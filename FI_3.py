import numpy as np
import time
import pandas as pd
from sklearn.datasets import make_moons, make_circles, load_digits, load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from scipy.spatial import ConvexHull

#############################################
# Fractal Generation
#############################################

def generate_fractal_points(transformations, num_points=500, seed=0):
    # transformations: array of shape (num_transformations, 6)
    # (a,b,c,d,e,f)
    np.random.seed(seed)
    num_transformations = transformations.shape[0]
    x, y = 0.0, 0.0
    points = []
    for _ in range(num_points):
        t = transformations[np.random.randint(num_transformations)]
        x_new = t[0]*x + t[1]*y + t[2]
        y_new = t[3]*x + t[4]*y + t[5]
        x, y = x_new, y_new
        points.append((x, y))
    return np.array(points)


#############################################
# Feature Extraction
#############################################

def fractal_dimension(points, box_sizes=[1.0, 2.0, 4.0, 8.0]):
    # Rough estimate of fractal dimension using box-counting on a grid.
    if len(points) == 0:
        return 0.0
    xs = points[:,0]
    ys = points[:,1]
    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)
    width = max_x - min_x + 1e-9
    height = max_y - min_y + 1e-9

    counts = []
    sizes = []
    for s in box_sizes:
        gx = int(np.ceil(width/s))
        gy = int(np.ceil(height/s))
        hist = np.zeros((gx, gy))
        for (X, Y) in points:
            ix = int((X - min_x)/s)
            iy = int((Y - min_y)/s)
            if 0 <= ix < gx and 0 <= iy < gy:
                hist[ix, iy] = 1
        N = np.sum(hist > 0)
        counts.append(N)
        sizes.append(s)

    # fractal dimension ~ slope of log(N) vs log(1/s)
    sizes = np.array(sizes)
    counts = np.array(counts)
    logN = np.log(counts+1e-9)
    logS = np.log(1.0/sizes)
    # linear fit
    A = np.vstack([logS, np.ones(len(logS))]).T
    m, c = np.linalg.lstsq(A, logN, rcond=None)[0]
    return m  # slope ~ fractal dimension

def lacunarity(points, box_size=5.0):
    # Simple lacunarity measure: compute variance/mean^2 of points per box in a grid.
    if len(points) == 0:
        return 0.0
    xs = points[:,0]
    ys = points[:,1]
    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)
    width = max_x - min_x + 1e-9
    height = max_y - min_y + 1e-9

    gx = max(1, int(width/box_size))
    gy = max(1, int(height/box_size))
    hist = np.zeros((gx, gy))
    for (X, Y) in points:
        ix = int((X - min_x)//box_size)
        iy = int((Y - min_y)//box_size)
        ix = min(ix, gx-1)
        iy = min(iy, gy-1)
        hist[ix, iy] += 1
    mean_val = np.mean(hist)
    var_val = np.var(hist)
    if mean_val > 1e-9:
        return var_val/(mean_val**2)
    else:
        return 0.0

def extract_features(points):
    # Extract geometric + fractal dimension + lacunarity features
    if len(points) < 3:
        # minimal fallback
        return np.zeros(12)

    xs = points[:,0]
    ys = points[:,1]

    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)
    width = xmax - xmin
    height = ymax - ymin
    x_mean = np.mean(xs)
    y_mean = np.mean(ys)
    x_var = np.var(xs)
    y_var = np.var(ys)

    try:
        hull = ConvexHull(points)
        hull_area = hull.area
    except:
        hull_area = 0.0

    fdim = fractal_dimension(points)
    lac = lacunarity(points)

    return np.array([width, height, x_mean, y_mean, x_var, y_var, hull_area, fdim, lac, len(points), np.mean(xs*ys), np.mean(np.abs(xs)+np.abs(ys))])


#############################################
# Genetic Programming Infrastructure
#############################################

# Function set and terminals for GP
# We'll define a simple symbolic expression:
# Terminals: meanX, stdX, varX, and constants
# Functions: +, -, *, sin, cos, tanh
import math, random

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

ALL_TERMINALS = TERMINALS + CONSTANTS

def generate_random_expr(max_depth=3):
    if max_depth == 0 or random.random() < 0.3:
        return str(random.choice(ALL_TERMINALS))
    # choose a function
    f, arity = random.choice(FUNCTIONS)
    if arity == 2:
        return (f, generate_random_expr(max_depth-1), generate_random_expr(max_depth-1))
    else:
        return (f, generate_random_expr(max_depth-1))

def expr_to_str(expr):
    if isinstance(expr, str):
        return expr
    if isinstance(expr, tuple):
        f = expr[0]
        if f in ['+', '-', '*']:
            return '('+expr_to_str(expr[1]) + f + expr_to_str(expr[2])+')'
        else:
            return f+'('+expr_to_str(expr[1])+')'
    return str(expr)

def evaluate_expr(expr, meanX, stdX, varX):
    if isinstance(expr, str):
        if expr == 'meanX':
            return meanX
        elif expr == 'stdX':
            return stdX
        elif expr == 'varX':
            return varX
        else:
            # constant
            return float(expr)
    f = expr[0]
    if f == '+':
        return evaluate_expr(expr[1], meanX, stdX, varX) + evaluate_expr(expr[2], meanX, stdX, varX)
    elif f == '-':
        return evaluate_expr(expr[1], meanX, stdX, varX) - evaluate_expr(expr[2], meanX, stdX, varX)
    elif f == '*':
        return evaluate_expr(expr[1], meanX, stdX, varX) * evaluate_expr(expr[2], meanX, stdX, varX)
    elif f == 'sin':
        return math.sin(evaluate_expr(expr[1], meanX, stdX, varX))
    elif f == 'cos':
        return math.cos(evaluate_expr(expr[1], meanX, stdX, varX))
    elif f == 'tanh':
        return math.tanh(evaluate_expr(expr[1], meanX, stdX, varX))
    return 0.0

def random_subtree(expr):
    # returns a random subtree (node)
    # We'll do a BFS collection of nodes
    nodes = []

    def collect(e):
        nodes.append(e)
        if isinstance(e, tuple):
            for c in e[1:]:
                collect(c)
    collect(expr)
    return random.choice(nodes)

def replace_subtree(expr, target, replacement):
    # replace a subtree equal to target with replacement
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
    # pick a random subtree and replace it
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
    # An individual is a tuple of 6 expressions for (a,b,c,d,e,f)
    return [generate_random_expr(3) for _ in range(1)]*3, [generate_random_expr(3) for _ in range(1)]*3
    # Actually we want num_transformations=3 each with 6 params:
    # Let's fix num_transformations=1 for simplicity due to complexity.
    # So we just generate one set of (a,b,c,d,e,f):
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

def evaluate_individual(ind, X_train, y_train, X_val, y_val, num_points=500):
    # Use a small subset for speed
    train_sample_size = min(50, len(X_train))
    val_sample_size = min(50, len(X_val))
    idx_train = np.random.choice(len(X_train), train_sample_size, replace=False)
    idx_val = np.random.choice(len(X_val), val_sample_size, replace=False)
    X_train_s = X_train[idx_train]
    y_train_s = y_train[idx_train]
    X_val_s = X_val[idx_val]
    y_val_s = y_val[idx_val]

    # For each sample, compute fractal parameters
    # ind is 6 expressions for (a,b,c,d,e,f)
    # Each param depends on meanX,stdX,varX
    # We'll construct transformations: shape (1,6)
    # Just 1 transformation for simplicity
    X_train_f = []
    X_val_f = []

    for X_ in X_train_s:
        meanX = np.mean(X_)
        stdX = np.std(X_)+1e-9
        varX = np.var(X_)+1e-9
        params = [evaluate_expr(expr, meanX, stdX, varX) for expr in ind]
        transformations = np.array(params).reshape(1,6)
        points = generate_fractal_points(transformations, num_points=num_points, seed=0)
        feats = extract_features(points)
        X_train_f.append(feats)

    for X_ in X_val_s:
        meanX = np.mean(X_)
        stdX = np.std(X_)+1e-9
        varX = np.var(X_)+1e-9
        params = [evaluate_expr(expr, meanX, stdX, varX) for expr in ind]
        transformations = np.array(params).reshape(1,6)
        points = generate_fractal_points(transformations, num_points=num_points, seed=0)
        feats = extract_features(points)
        X_val_f.append(feats)

    X_train_f = np.array(X_train_f)
    X_val_f = np.array(X_val_f)

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train_f, y_train_s)
    preds = clf.predict(X_val_f)
    return balanced_accuracy_score(y_val_s, preds)

def tournament_selection(pop, fitnesses, k=3):
    # pick k random individuals and return the best
    selected = random.sample(list(zip(pop,fitnesses)), k)
    selected.sort(key=lambda x:x[1], reverse=True)
    return selected[0][0]

#############################################
# Run the Evolutionary Process
#############################################

def run_evolution(X_train, y_train, X_val, y_val, pop_size=10, generations=10):
    # Initialize population
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
        # If pop_size is odd, just add one more random
        if len(new_pop) < pop_size:
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
    # Train final classifier on full train+val
    X_trainval_f = []
    X_test_f = []
    for X_ in X_trainval:
        meanX = np.mean(X_)
        stdX = np.std(X_)+1e-9
        varX = np.var(X_)+1e-9
        params = [evaluate_expr(expr, meanX, stdX, varX) for expr in best_ind]
        transformations = np.array(params).reshape(1,6)
        points = generate_fractal_points(transformations, num_points=500, seed=0)
        feats = extract_features(points)
        X_trainval_f.append(feats)
    for X_ in X_test:
        meanX = np.mean(X_)
        stdX = np.std(X_)+1e-9
        varX = np.var(X_)+1e-9
        params = [evaluate_expr(expr, meanX, stdX, varX) for expr in best_ind]
        transformations = np.array(params).reshape(1,6)
        points = generate_fractal_points(transformations, num_points=500, seed=0)
        feats = extract_features(points)
        X_test_f.append(feats)

    X_trainval_f = np.array(X_trainval_f)
    X_test_f = np.array(X_test_f)

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_trainval_f, y_trainval)
    preds = clf.predict(X_test_f)
    return balanced_accuracy_score(y_test, preds)


def test_on_dataset_fi(name: str, X: np.ndarray, y: np.ndarray):
    # Train/Val/Test split: 60/20/20
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_trainval = scaler.transform(X_trainval)
    X_test = scaler.transform(X_test)

    start_time = time.time()
    best_ind, best_fit = run_evolution(X_train, y_train, X_val, y_val, pop_size=10, generations=5)
    train_time = time.time() - start_time

    start_time = time.time()
    acc = final_evaluation(best_ind, X_trainval, y_trainval, X_test, y_test)
    test_time = time.time() - start_time
    print(f"Final Balanced Accuracy: {acc:.4f}")
    return acc, train_time, test_time


def run_experiments_fi():
    datasets = {
        "Moons": lambda: make_moons(n_samples=500, noise=0.2, random_state=42),
        "Circles": lambda: make_circles(n_samples=500, noise=0.2, factor=0.5, random_state=42),
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
