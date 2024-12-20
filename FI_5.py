import numpy as np
import time
import pandas as pd
import math, random
from sklearn.datasets import make_moons, make_circles, load_digits, load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from numba import njit

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Parameters
NUM_POINTS = 200
SCALES = [50, 100, 200]  # partial patterns
PARAM_LIMIT = 10.0
POP_SIZE = 5
GENERATIONS = 3
GRID_SIZE = 50  # fixed-size grid to avoid memory issues
BOX_SIZES = np.array([2.0,4.0,8.0], dtype=np.float64) # for fractal dimension approximation

FUNCTIONS = [
    ('+', 2),
    ('-', 2),
    ('*', 2),
    ('sin', 1),
    ('cos', 1),
    ('tanh',1),
]
TERMINALS = ['meanX', 'stdX', 'varX']
CONSTANTS = [0.1, 0.2, 0.5, 1.0, 2.0, -0.1, -0.5]
ALL_TERMINALS = TERMINALS + [str(c) for c in CONSTANTS]

def generate_random_expr(max_depth=3):
    if max_depth == 0 or random.random() < 0.3:
        return str(random.choice(ALL_TERMINALS))
    f, arity = random.choice(FUNCTIONS)
    if arity == 2:
        return (f, generate_random_expr(max_depth-1), generate_random_expr(max_depth-1))
    else:
        return (f, generate_random_expr(max_depth-1))

def evaluate_expr(expr, meanX, stdX, varX):
    if isinstance(expr, str):
        if expr == 'meanX':
            return meanX
        elif expr == 'stdX':
            return stdX
        elif expr == 'varX':
            return varX
        else:
            return float(expr)
    f = expr[0]
    if f in ['+', '-', '*']:
        a1 = evaluate_expr(expr[1], meanX, stdX, varX)
        a2 = evaluate_expr(expr[2], meanX, stdX, varX)
        if f == '+': return a1+a2
        elif f == '-': return a1-a2
        else: return a1*a2
    else:
        a1 = evaluate_expr(expr[1], meanX, stdX, varX)
        if f == 'sin':
            return math.sin(a1)
        elif f == 'cos':
            return math.cos(a1)
        elif f == 'tanh':
            return math.tanh(a1)
    return 0.0

def random_subtree(expr):
    nodes = []
    def collect(e):
        nodes.append(e)
        if isinstance(e, tuple):
            for c in e[1:]:
                collect(c)
    collect(expr)
    return random.choice(nodes)

def replace_subtree(expr, target, replacement):
    if expr is target:
        return replacement
    if isinstance(expr, tuple):
        if len(expr) == 2:
            return (expr[0], replace_subtree(expr[1], target, replacement))
        else:
            return (expr[0],
                    replace_subtree(expr[1], target, replacement),
                    replace_subtree(expr[2], target, replacement))
    return expr

def mutate(expr, max_depth=3):
    t = random_subtree(expr)
    new_sub = generate_random_expr(max_depth)
    return replace_subtree(expr, t, new_sub)

def crossover(e1, e2):
    t1 = random_subtree(e1)
    t2 = random_subtree(e2)
    e1_new = replace_subtree(e1, t1, t2)
    e2_new = replace_subtree(e2, t2, t1)
    return e1_new, e2_new

def copy_expr(expr):
    if isinstance(expr, str):
        return expr
    if len(expr) == 2:
        return (expr[0], copy_expr(expr[1]))
    else:
        return (expr[0], copy_expr(expr[1]), copy_expr(expr[2]))

def generate_individual():
    return [generate_random_expr(3) for _ in range(6)]

def copy_individual(ind):
    return [copy_expr(e) for e in ind]

def mutate_ind(ind):
    i = random.randint(0, len(ind)-1)
    ind[i] = mutate(ind[i], 3)
    return ind

def crossover_ind(ind1, ind2):
    i = random.randint(0, len(ind1)-1)
    j = random.randint(0, len(ind2)-1)
    c1, c2 = crossover(ind1[i], ind2[j])
    ind1_new = copy_individual(ind1)
    ind2_new = copy_individual(ind2)
    ind1_new[i] = c1
    ind2_new[j] = c2
    return ind1_new, ind2_new

def clamp_params(params):
    return np.clip(params, -PARAM_LIMIT, PARAM_LIMIT)

@njit
def generate_fractal_points(transformations, num_points=200, seed=0):
    np.random.seed(seed)
    x, y = 0.0, 0.0
    points = np.empty((num_points, 2), dtype=np.float64)
    t_count = transformations.shape[0]
    count=0
    for i in range(num_points):
        idx = np.random.randint(t_count)
        t = transformations[idx]
        x_new = t[0]*x + t[1]*y + t[2]
        y_new = t[3]*x + t[4]*y + t[5]
        if not (math.isfinite(x_new) and math.isfinite(y_new)):
            break
        if abs(x_new)>1e6 or abs(y_new)>1e6:
            break
        x, y = x_new, y_new
        points[count,0] = x
        points[count,1] = y
        count+=1
    return points[:count]

@njit
def to_grid(points, grid_size=GRID_SIZE):
    # Scale points to [0,1] and map to grid
    if points.shape[0]<1:
        return np.zeros((grid_size,grid_size), dtype=np.float64)
    xs = points[:,0]
    ys = points[:,1]
    xmin = np.min(xs)
    xmax = np.max(xs)
    ymin = np.min(ys)
    ymax = np.max(ys)
    width = (xmax - xmin)+1e-9
    height = (ymax - ymin)+1e-9
    grid = np.zeros((grid_size,grid_size), dtype=np.float64)
    for i in range(points.shape[0]):
        gx = int((points[i,0]-xmin)/width*grid_size)
        gy = int((points[i,1]-ymin)/height*grid_size)
        if gx>=0 and gx<grid_size and gy>=0 and gy<grid_size:
            grid[gx,gy]+=1
    return grid

@njit
def fractal_dimension(grid, box_sizes=np.array([2.0,4.0,8.0], dtype=np.float64)):
    # Approx fractal dimension via box-counting on the fixed grid
    gx, gy = grid.shape
    counts = np.empty(box_sizes.size, dtype=np.float64)
    for i in range(box_sizes.size):
        s = box_sizes[i]
        step_x = int(gx/s)
        step_y = int(gy/s)
        if step_x<1 or step_y<1:
            counts[i]=1e-9
            continue
        sub_count=0
        for xx in range(0,gx,step_x):
            for yy in range(0,gy,step_y):
                # check if any point in this box
                found=False
                for u in range(xx, min(xx+step_x, gx)):
                    for v in range(yy, min(yy+step_y, gy)):
                        if grid[u,v]>0:
                            found=True
                            break
                    if found:
                        break
                if found:
                    sub_count+=1
        if sub_count<1:
            sub_count=1e-9
        counts[i]=sub_count
    logN = np.log(counts)
    scale = box_sizes
    logS = np.log(1.0/scale)

    # compute slope m:
    meanX = 0.0
    meanY = 0.0
    n = logS.size
    for val in logS:
        meanX += val
    meanX/=n
    for val in logN:
        meanY+=val
    meanY/=n
    num=0.0
    den=0.0
    for j in range(n):
        dx = logS[j]-meanX
        dy = logN[j]-meanY
        num += dx*dy
        den += dx*dx
    if den<1e-12:
        return 0.0
    m = num/den
    return m

@njit
def lacunarity(grid):
    # Lacunarity: var/mean^2 of box counts
    mean_val = 0.0
    count=0
    for row in grid:
        for val in row:
            mean_val+=val
            count+=1
    if count<1:
        return 0.0
    mean_val /= count
    var_val=0.0
    for row in grid:
        for val in row:
            diff = val-mean_val
            var_val += diff*diff
    var_val/=count
    if mean_val>1e-12:
        return var_val/(mean_val**2)
    else:
        return 0.0

@njit
def basic_geometry(grid):
    # from grid get width,height,x_var,y_var approximations
    # approximate coordinates from grid
    gx,gy = grid.shape
    total = 0.0
    sumx=0.0
    sumy=0.0
    for x in range(gx):
        for y in range(gy):
            w = grid[x,y]
            total+=w
            sumx+=x*w
            sumy+=y*w
    if total<1e-9:
        return (0.0,0.0,0.0,0.0)
    meanx = sumx/total
    meany = sumy/total
    varx=0.0
    vary=0.0
    minx = gx
    maxx = 0
    miny = gy
    maxy = 0
    for x in range(gx):
        for y in range(gy):
            w=grid[x,y]
            if w>0:
                if x<minx:
                    minx=x
                if x>maxx:
                    maxx=x
                if y<miny:
                    miny=y
                if y>maxy:
                    maxy=y
                dx = x-meanx
                dy = y-meany
                varx+=dx*dx*w
                vary+=dy*dy*w
    varx/=total
    vary/=total
    width = (maxx-minx+1e-9)/(gx)
    height=(maxy-miny+1e-9)/(gy)
    return (width,height,varx,vary)

@njit
def extract_features_single_scale(points):
    # If not enough points
    if points.shape[0]<2:
        return np.zeros(6, dtype=np.float64)
    grid = to_grid(points, GRID_SIZE)
    fd = fractal_dimension(grid)
    lac = lacunarity(grid)
    w,h,vx,vy = basic_geometry(grid)
    # 6 features: fd, lac, width, height, varx, vary
    feats = np.array([fd,lac,w,h,vx,vy], dtype=np.float64)
    return feats

@njit
def extract_features_multi_scale(points):
    # For each scale in SCALES, compute features_single_scale and concat
    # SCALES * 6 features total
    feat_len = len(SCALES)*6
    feats = np.zeros(feat_len, dtype=np.float64)
    for i,sc in enumerate(SCALES):
        if points.shape[0]>=sc:
            sub = points[:sc]
            f = extract_features_single_scale(sub)
            for j in range(6):
                feats[i*6+j]=f[j]
        else:
            # no enough points, keep zeros
            pass
    return feats

def evaluate_individual(ind, X_train, y_train, X_val, y_val):
    train_sample_size = min(30, len(X_train))
    val_sample_size = min(30, len(X_val))
    idx_train = np.random.choice(len(X_train), train_sample_size, replace=False)
    idx_val = np.random.choice(len(X_val), val_sample_size, replace=False)
    X_train_s = X_train[idx_train]
    y_train_s = y_train[idx_train]
    X_val_s = X_val[idx_val]
    y_val_s = y_val[idx_val]

    feat_len = len(SCALES)*6
    X_train_f = np.empty((train_sample_size, feat_len))
    X_val_f = np.empty((val_sample_size, feat_len))

    for i, sample in enumerate(X_train_s):
        meanX = np.mean(sample)
        stdX = np.std(sample)+1e-9
        varX = np.var(sample)+1e-9
        params = [evaluate_expr(expr, meanX, stdX, varX) for expr in ind]
        params = clamp_params(params)
        transformations = np.array(params).reshape(1,6)
        points = generate_fractal_points(transformations, num_points=NUM_POINTS, seed=SEED)
        if len(points)>0:
            feats = extract_features_multi_scale(points)
        else:
            feats = np.zeros(feat_len)
        X_train_f[i] = feats

    for i, sample in enumerate(X_val_s):
        meanX = np.mean(sample)
        stdX = np.std(sample)+1e-9
        varX = np.var(sample)+1e-9
        params = [evaluate_expr(expr, meanX, stdX, varX) for expr in ind]
        params = clamp_params(params)
        transformations = np.array(params).reshape(1,6)
        points = generate_fractal_points(transformations, num_points=NUM_POINTS, seed=SEED)
        if len(points)>0:
            feats = extract_features_multi_scale(points)
        else:
            feats = np.zeros(feat_len)
        X_val_f[i] = feats

    clf = DecisionTreeClassifier(random_state=SEED)
    clf.fit(X_train_f, y_train_s)
    preds = clf.predict(X_val_f)
    return balanced_accuracy_score(y_val_s, preds)

def tournament_selection(pop, fitnesses, k=3):
    selected = random.sample(list(zip(pop,fitnesses)), k)
    selected.sort(key=lambda x:x[1], reverse=True)
    return selected[0][0]

def run_evolution(X_train, y_train, X_val, y_val, pop_size=POP_SIZE, generations=GENERATIONS):
    population = [generate_individual() for _ in range(pop_size)]
    fitnesses = [evaluate_individual(ind, X_train, y_train, X_val, y_val) for ind in population]
    best_fit = max(fitnesses)
    best_ind = population[np.argmax(fitnesses)]
    print(f"Initial best fitness: {best_fit:.4f}")

    for g in range(generations):
        new_pop = []
        for _ in range(pop_size//2):
            p1 = tournament_selection(population, fitnesses, k=3)
            p2 = tournament_selection(population, fitnesses, k=3)
            c1, c2 = crossover_ind(p1,p2)
            if random.random()<0.3:
                c1 = mutate_ind(c1)
            if random.random()<0.3:
                c2 = mutate_ind(c2)
            new_pop.append(c1)
            new_pop.append(c2)
        if len(new_pop)<pop_size:
            new_pop.append(generate_individual())
        population = new_pop
        fitnesses = [evaluate_individual(ind, X_train, y_train, X_val, y_val) for ind in population]
        gen_best_fit = max(fitnesses)
        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_ind = population[np.argmax(fitnesses)]
        print(f"Generation {g}, Best Fitness: {gen_best_fit:.4f}, Overall Best: {best_fit:.4f}")

    return best_ind, best_fit

def final_evaluation(best_ind, X_trainval, y_trainval, X_test, y_test):
    feat_len = len(SCALES)*6
    X_trainval_f = np.empty((len(X_trainval), feat_len))
    X_test_f = np.empty((len(X_test), feat_len))

    for i, sample in enumerate(X_trainval):
        meanX = np.mean(sample)
        stdX = np.std(sample)+1e-9
        varX = np.var(sample)+1e-9
        params = [evaluate_expr(expr, meanX, stdX, varX) for expr in best_ind]
        params = clamp_params(params)
        transformations = np.array(params).reshape(1,6)
        points = generate_fractal_points(transformations, num_points=NUM_POINTS, seed=SEED)
        if len(points)>0:
            feats = extract_features_multi_scale(points)
        else:
            feats = np.zeros(feat_len)
        X_trainval_f[i] = feats

    for i, sample in enumerate(X_test):
        meanX = np.mean(sample)
        stdX = np.std(sample)+1e-9
        varX = np.var(sample)+1e-9
        params = [evaluate_expr(expr, meanX, stdX, varX) for expr in best_ind]
        params = clamp_params(params)
        transformations = np.array(params).reshape(1,6)
        points = generate_fractal_points(transformations, num_points=NUM_POINTS, seed=SEED)
        if len(points)>0:
            feats = extract_features_multi_scale(points)
        else:
            feats = np.zeros(feat_len)
        X_test_f[i] = feats

    clf = DecisionTreeClassifier(random_state=SEED)
    clf.fit(X_trainval_f, y_trainval)
    preds = clf.predict(X_test_f)
    return balanced_accuracy_score(y_test, preds)

def test_on_dataset_fi(name: str, X: np.ndarray, y: np.ndarray):
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=SEED, stratify=y_trainval)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_trainval = scaler.transform(X_trainval)
    X_test = scaler.transform(X_test)

    start_time = time.time()
    best_ind, best_fit = run_evolution(X_train, y_train, X_val, y_val)
    train_time = time.time() - start_time

    start_time = time.time()
    acc = final_evaluation(best_ind, X_trainval, y_trainval, X_test, y_test)
    test_time = time.time() - start_time
    print(f"Final Balanced Accuracy: {acc:.4f}")
    return acc, train_time, test_time

def run_experiments_fi():
    datasets = {
        "Moons": lambda: make_moons(n_samples=500, noise=0.2, random_state=SEED),
        "Circles": lambda: make_circles(n_samples=500, noise=0.2, factor=0.5, random_state=SEED),
        "Digits": lambda: load_digits(return_X_y=True),
        "Iris": lambda: load_iris(return_X_y=True),
        "Breast Cancer": lambda: load_breast_cancer(return_X_y=True),
        "Wine": lambda: load_wine(return_X_y=True),
    }

    fi_results = []
    for dataset_name, dataset_loader in datasets.items():
        print(f"\nTesting FI on {dataset_name} dataset...")
        X, y = dataset_loader()
        fi_acc, fi_train_time, fi_pred_time = test_on_dataset_fi(dataset_name, X, y)
        fi_results.append({
            "Dataset": dataset_name,
            "FI Balanced Accuracy": fi_acc,
            "FI Training Time (s)": round(fi_train_time, 4),
            "FI Prediction Time (s)": round(fi_pred_time, 4),
        })

    print("\n" + "=" * 80)
    print("SUMMARY OF FI RESULTS")
    print("=" * 80)
    fi_results_df = pd.DataFrame(fi_results)
    print(fi_results_df.to_string(index=False))

if __name__ == "__main__":
    run_experiments_fi()
