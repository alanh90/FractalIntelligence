'''
================================================================================
SUMMARY OF FI RESULTS
================================================================================
      Dataset  FI Balanced Accuracy  FI Total Training Time (s)  FI Prediction Time (s)
        Moons              0.660000                      1.1702                  0.0170
      Circles              0.690000                      0.0567                  0.0160
       Digits              0.114114                      0.1085                  0.0833
         Iris              0.733333                      0.0694                  0.0070
Breast Cancer              0.871032                      0.0658                  0.0218
         Wine              0.250000                      0.0636                  0.0070
'''

import numpy as np
import time
import pandas as pd
import math, random
from sklearn.datasets import make_moons, make_circles, load_digits, load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from numba import njit

##########################################################
# Parameters
##########################################################
NUM_POINTS = 200   # number of fractal points
POP_SIZE = 5
GENERATIONS = 3
PARAM_LIMIT = 10.0
GRID_SIZE = 10    # for 2D histogram
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

##########################################################
# Utility functions
##########################################################

@njit
def make_finite(arr):
    # Replace any NaN or infinite values with 0.0
    for i in range(arr.size):
        val = arr[i]
        if not math.isfinite(val):
            arr[i] = 0.0
    return arr

##########################################################
# Fractal Generation with Numba
##########################################################

@njit
def generate_fractal_points(transformations, num_points=200, seed=0):
    np.random.seed(seed)
    x, y = 0.0, 0.0
    points = np.empty((num_points, 2), dtype=np.float64)
    t_count = transformations.shape[0]
    for i in range(num_points):
        idx = np.random.randint(t_count)
        t = transformations[idx]
        x_new = t[0]*x + t[1]*y + t[2]
        y_new = t[3]*x + t[4]*y + t[5]
        # Clamp coordinates
        if not (math.isfinite(x_new) and math.isfinite(y_new)):
            return np.empty((0,2))
        if abs(x_new)>1e6 or abs(y_new)>1e6:
            return np.empty((0,2))
        x, y = x_new, y_new
        points[i,0] = x
        points[i,1] = y
    return points

@njit
def extract_features(points, grid_size=GRID_SIZE):
    # If no points, return zeros
    feat_len = 6 + grid_size*grid_size
    if points.shape[0]<3:
        return np.zeros(feat_len, dtype=np.float64)

    xs = points[:,0]
    ys = points[:,1]
    # Check if finite
    for i in range(xs.size):
        if not math.isfinite(xs[i]) or not math.isfinite(ys[i]):
            return np.zeros(feat_len, dtype=np.float64)

    xmin = np.min(xs)
    xmax = np.max(xs)
    ymin = np.min(ys)
    ymax = np.max(ys)
    width = (xmax - xmin) + 1e-9
    height = (ymax - ymin) + 1e-9

    x_mean = np.mean(xs)
    y_mean = np.mean(ys)
    x_var = np.var(xs)
    y_var = np.var(ys)

    # Create histogram
    hist = np.zeros(grid_size*grid_size, dtype=np.float64)
    for i in range(xs.size):
        gx = int((xs[i]-xmin)/width * grid_size)
        gy = int((ys[i]-ymin)/height * grid_size)
        if gx>=0 and gx<grid_size and gy>=0 and gy<grid_size:
            hist[gx*grid_size + gy]+=1

    hist_sum = 0.0
    for v in hist:
        hist_sum += v

    if hist_sum > 1e-12:
        for i in range(hist.size):
            hist[i] = hist[i]/hist_sum
    else:
        for i in range(hist.size):
            hist[i] = 0.0

    feats = np.zeros(feat_len, dtype=np.float64)
    feats[0] = width
    feats[1] = height
    feats[2] = x_mean
    feats[3] = y_mean
    feats[4] = x_var
    feats[5] = y_var
    for i in range(hist.size):
        feats[6+i] = hist[i]

    feats = make_finite(feats)
    return feats

##########################################################
# Genetic Programming Setup
##########################################################

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

def evaluate_individual(ind, X_train, y_train, X_val, y_val):
    train_sample_size = min(30, len(X_train))
    val_sample_size = min(30, len(X_val))
    idx_train = np.random.choice(len(X_train), train_sample_size, replace=False)
    idx_val = np.random.choice(len(X_val), val_sample_size, replace=False)
    X_train_s = X_train[idx_train]
    y_train_s = y_train[idx_train]
    X_val_s = X_val[idx_val]
    y_val_s = y_val[idx_val]

    feat_len = 6 + GRID_SIZE*GRID_SIZE
    X_train_f = np.empty((train_sample_size, feat_len))
    X_val_f = np.empty((val_sample_size, feat_len))

    for i, X_ in enumerate(X_train_s):
        meanX = np.mean(X_)
        stdX = np.std(X_)+1e-9
        varX = np.var(X_)+1e-9
        params = [evaluate_expr(expr, meanX, stdX, varX) for expr in ind]
        params = clamp_params(params)
        transformations = np.array(params).reshape(1,6)
        points = generate_fractal_points(transformations, num_points=NUM_POINTS, seed=0)
        if points.shape[0]>0:
            feats = extract_features(points)
        else:
            feats = np.zeros(feat_len)
        X_train_f[i] = feats

    for i, X_ in enumerate(X_val_s):
        meanX = np.mean(X_)
        stdX = np.std(X_)+1e-9
        varX = np.var(X_)+1e-9
        params = [evaluate_expr(expr, meanX, stdX, varX) for expr in ind]
        params = clamp_params(params)
        transformations = np.array(params).reshape(1,6)
        points = generate_fractal_points(transformations, num_points=NUM_POINTS, seed=0)
        if points.shape[0]>0:
            feats = extract_features(points)
        else:
            feats = np.zeros(feat_len)
        X_val_f[i] = feats

    # No nan_to_num with posinf, neginf in Numba, we already avoided that
    # Just ensure finite:
    X_train_f = make_finite(X_train_f.ravel()).reshape(X_train_f.shape)
    X_val_f = make_finite(X_val_f.ravel()).reshape(X_val_f.shape)

    clf = LinearSVC(max_iter=1000, random_state=SEED)
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
    feat_len = 6 + GRID_SIZE*GRID_SIZE
    X_trainval_f = np.empty((len(X_trainval), feat_len))
    X_test_f = np.empty((len(X_test), feat_len))

    for i, X_ in enumerate(X_trainval):
        meanX = np.mean(X_)
        stdX = np.std(X_)+1e-9
        varX = np.var(X_)+1e-9
        params = [evaluate_expr(expr, meanX, stdX, varX) for expr in best_ind]
        params = clamp_params(params)
        transformations = np.array(params).reshape(1,6)
        points = generate_fractal_points(transformations, num_points=NUM_POINTS, seed=0)
        if points.shape[0]>0:
            feats = extract_features(points)
        else:
            feats = np.zeros(feat_len)
        X_trainval_f[i] = feats

    for i, X_ in enumerate(X_test):
        meanX = np.mean(X_)
        stdX = np.std(X_)+1e-9
        varX = np.var(X_)+1e-9
        params = [evaluate_expr(expr, meanX, stdX, varX) for expr in best_ind]
        params = clamp_params(params)
        transformations = np.array(params).reshape(1,6)
        points = generate_fractal_points(transformations, num_points=NUM_POINTS, seed=0)
        if points.shape[0]>0:
            feats = extract_features(points)
        else:
            feats = np.zeros(feat_len)
        X_test_f[i] = feats

    X_trainval_f = make_finite(X_trainval_f.ravel()).reshape(X_trainval_f.shape)
    X_test_f = make_finite(X_test_f.ravel()).reshape(X_test_f.shape)

    clf = LinearSVC(max_iter=1000, random_state=SEED)
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
