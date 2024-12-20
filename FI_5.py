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

GRID_SIZE = 50   # fixed grid size for dimension & lacunarity
BOX_SIZES = [2,4,8]  # few box sizes for fractal dimension

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
    points = np.empty((num_points,2), dtype=np.float64)
    t_count = transformations.shape[0]
    length = 0
    for i in range(num_points):
        idx = np.random.randint(t_count)
        t = transformations[idx]
        x_new = t[0]*x + t[1]*y + t[2]
        y_new = t[3]*x + t[4]*y + t[5]
        if not (np.isfinite(x_new) and np.isfinite(y_new)):
            break
        if abs(x_new)>1e6 or abs(y_new)>1e6:
            break
        x, y = x_new, y_new
        points[i,0] = x
        points[i,1] = y
        length+=1
    return points[:length]

@njit
def safe_scale_points(points, grid_size=GRID_SIZE):
    if points.shape[0]<2:
        return False, np.zeros((grid_size,grid_size))
    xs = points[:,0]
    ys = points[:,1]
    if xs.size<2:
        return False, np.zeros((grid_size,grid_size))
    xmin = np.min(xs)
    xmax = np.max(xs)
    ymin = np.min(ys)
    ymax = np.max(ys)
    width = xmax - xmin + 1e-9
    height = ymax - ymin + 1e-9
    grid = np.zeros((grid_size,grid_size), dtype=np.float64)
    for i in range(xs.size):
        X = (xs[i]-xmin)/width
        Y = (ys[i]-ymin)/height
        if 0<=X<1 and 0<=Y<1:
            gx = int(X*grid_size)
            gy = int(Y*grid_size)
            grid[gx,gy]+=1
    return True, grid

@njit
def fractal_dimension(grid, box_sizes=(2,4,8)):
    # box-count dimension on scaled grid
    gx, gy = grid.shape
    counts = []
    scale_factors = []
    for b in box_sizes:
        if b>0:
            step_x = gx//b
            step_y = gy//b
            if step_x<1 or step_y<1:
                # too large a box partition, skip
                continue
            boxes = 0
            for ix in range(b):
                for iy in range(b):
                    sub = grid[ix*step_x:(ix+1)*step_x, iy*step_y:(iy+1)*step_y]
                    if np.any(sub>0):
                        boxes+=1
            counts.append(boxes)
            scale_factors.append(b)
    if len(counts)<2:
        return 0.0
    counts = np.array(counts, dtype=np.float64)
    scale_factors = np.array(scale_factors, dtype=np.float64)
    logN = np.log(counts+1e-9)
    logS = np.log(1.0/scale_factors)
    A = np.vstack((logS, np.ones(len(logS)))).T
    sol = np.linalg.lstsq(A, logN, rcond=None)[0]
    m = sol[0]
    return m

@njit
def lacunarity(grid):
    # lacunarity = var/mean^2 of occupancy
    mean_val = np.mean(grid)
    var_val = np.var(grid)
    if mean_val>1e-9:
        return var_val/(mean_val**2)
    return 0.0

@njit
def basic_geometry(grid):
    # interpret grid as distribution of points
    gx, gy = grid.shape
    total = np.sum(grid)
    if total<1:
        return (0.0,0.0,0.0,0.0) # no structure
    xs = []
    ys = []
    for ix in range(gx):
        for iy in range(gy):
            c = grid[ix,iy]
            for _ in range(int(c)):
                xs.append(ix)
                ys.append(iy)
    if len(xs)<2:
        return (0.0,0.0,0.0,0.0)
    xarr = np.array(xs, dtype=np.float64)
    yarr = np.array(ys, dtype=np.float64)
    width = (np.max(xarr)-np.min(xarr)+1e-9)
    height = (np.max(yarr)-np.min(yarr)+1e-9)
    x_var = np.var(xarr)
    y_var = np.var(yarr)
    return (width,height,x_var,y_var)

@njit
def extract_features_single_scale(points):
    # scale and check
    ok, grid = safe_scale_points(points)
    if not ok:
        # no points or invalid, return zeros
        return np.zeros(6, dtype=np.float64)
    fd = fractal_dimension(grid)
    lac = lacunarity(grid)
    w,h,xv,yv = basic_geometry(grid)
    # fd, lac, width, height, x_var, y_var
    feats = np.array([fd, lac, w, h, xv, yv], dtype=np.float64)
    return feats

def extract_features_multi_scale(points):
    feat_list = []
    for scale in SCALES:
        if len(points)>=scale:
            psub = points[:scale]
            feats = extract_features_single_scale(psub)
        else:
            feats = np.zeros(6)
        feat_list.append(feats)
    return np.concatenate(feat_list)

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
        points = generate_fractal_points(transformations, num_points=NUM_POINTS, seed=0)
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
        points = generate_fractal_points(transformations, num_points=NUM_POINTS, seed=0)
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
        points = generate_fractal_points(transformations, num_points=NUM_POINTS, seed=0)
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
        points = generate_fractal_points(transformations, num_points=NUM_POINTS, seed=0)
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
            "FI Total Training Time (s)": round(fi_train_time, 4),
            "FI Prediction Time (s)": round(fi_pred_time, 4),
        })

    print("\n" + "=" * 80)
    print("SUMMARY OF FI RESULTS")
    print("=" * 80)
    fi_results_df = pd.DataFrame(fi_results)
    print(fi_results_df.to_string(index=False))

if __name__ == "__main__":
    run_experiments_fi()
