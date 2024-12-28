"""
===== Loading Breast Cancer Dataset =====

Dataset size: (569, 30), #labels=569

===== Building Fractal Mixture-of-Experts Model =====

Performing KMeans with n_clusters=3...
  - Cluster 0 size = 287
  - Cluster 1 size = 31
  - Cluster 2 size = 80

--- Evolving fractal for cluster 0 ---
Cluster 0: best fractal fitness after evolution = 0.5000
  => Local classifier cluster-balanced-acc = 0.5000

--- Evolving fractal for cluster 1 ---
Cluster 1 => trivial cluster => random fractal, no local classifier.

--- Evolving fractal for cluster 2 ---
Cluster 2: best fractal fitness after evolution = 0.5000
  => Local classifier cluster-balanced-acc = 0.5000

Training completed in 3.8709 seconds.

===== Predicting on Test Set =====


===== Final Results =====
Training Time   = 3.8709 seconds
Prediction Time = 0.1985 seconds
Balanced Accuracy on Test Set = 0.8454
============================================
"""

import numpy as np
import math
import random
import time
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from numba import njit

# -----------------------------
# Global Settings
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Mixture-of-Experts
N_CLUSTERS = 3

# Evolution parameters
POP_SIZE = 20
GENERATIONS = 10

# Fractal param limit (clamping)
PARAM_LIMIT = 6.0

# Fractal generation
NUM_FRAC_POINTS = 200   # how many points to generate
GRID_SIZE = 40          # for fractal dimension grid
BOX_SIZES = np.array([2.0, 4.0, 8.0], dtype=np.float64)

# We will do up to 'max_depth' expansions in an "adaptive" sense if we want
MAX_ITER_DEPTH = 2  # keep it small to avoid long runs; you can increase if desired

VERBOSE_PREDICT = False  # set True to see logs for each test sample


# -----------------------------
# Fractal & Feature Functions
# -----------------------------

def clamp_params(params):
    """Clamp an array of params to [-PARAM_LIMIT, PARAM_LIMIT]."""
    return np.clip(params, -PARAM_LIMIT, PARAM_LIMIT)

@njit
def generate_fractal_points(transformations, num_points, seed=0):
    """
    Generate fractal points using an Iterated Function System approach.
    Each row in 'transformations' is [a, b, e, c, d, f], so:
      x' = a*x + b*y + e
      y' = c*x + d*y + f
    """
    np.random.seed(seed)
    x, y = 0.0, 0.0
    t_count = transformations.shape[0]
    points = np.empty((num_points, 2), dtype=np.float64)
    idx_count = 0

    for i in range(num_points):
        idx = np.random.randint(t_count)
        t = transformations[idx]
        x_new = t[0] * x + t[1] * y + t[2]
        y_new = t[3] * x + t[4] * y + t[5]

        if not (math.isfinite(x_new) and math.isfinite(y_new)):
            break
        if abs(x_new) > 1e6 or abs(y_new) > 1e6:
            break

        x, y = x_new, y_new
        points[idx_count, 0] = x
        points[idx_count, 1] = y
        idx_count += 1
    return points[:idx_count]

@njit
def to_grid(points, grid_size):
    """
    Map points to a grid for fractal dimension, lacunarity, etc.
    """
    grid = np.zeros((grid_size, grid_size), dtype=np.float64)
    if points.shape[0] < 1:
        return grid

    xs = points[:, 0]
    ys = points[:, 1]
    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)

    width = (xmax - xmin) + 1e-9
    height = (ymax - ymin) + 1e-9

    for i in range(points.shape[0]):
        gx = int((points[i, 0] - xmin) / width * grid_size)
        gy = int((points[i, 1] - ymin) / height * grid_size)
        if 0 <= gx < grid_size and 0 <= gy < grid_size:
            grid[gx, gy] += 1
    return grid

@njit
def fractal_dimension(grid, box_sizes):
    """
    Approximate fractal dimension via box-counting.
    """
    gx, gy = grid.shape
    counts = np.empty(box_sizes.size, dtype=np.float64)

    for i in range(box_sizes.size):
        s = box_sizes[i]
        step_x = int(gx / s)
        step_y = int(gy / s)
        if step_x < 1 or step_y < 1:
            counts[i] = 1e-9
            continue

        sub_count = 0
        for xx in range(0, gx, step_x):
            for yy in range(0, gy, step_y):
                found = False
                for u in range(xx, min(xx + step_x, gx)):
                    for v in range(yy, min(yy + step_y, gy)):
                        if grid[u, v] > 0:
                            found = True
                            break
                    if found:
                        break
                if found:
                    sub_count += 1
        if sub_count < 1:
            sub_count = 1e-9
        counts[i] = sub_count

    logN = np.log(counts)
    logS = np.log(1.0 / box_sizes)

    meanX = 0.0
    meanY = 0.0
    n = logS.size
    for val in logS:
        meanX += val
    meanX /= n
    for val in logN:
        meanY += val
    meanY /= n

    num = 0.0
    den = 0.0
    for j in range(n):
        dx = logS[j] - meanX
        dy = logN[j] - meanY
        num += dx * dy
        den += dx * dx
    if den < 1e-12:
        return 0.0
    return num / den

@njit
def lacunarity(grid):
    """
    Lacunarity: var/mean^2 of box counts in the grid.
    """
    gx, gy = grid.shape
    total = 0.0
    count = 0
    for x in range(gx):
        for y in range(gy):
            total += grid[x, y]
            count += 1
    if count < 1:
        return 0.0

    mean_val = total / count
    var_val = 0.0
    for x in range(gx):
        for y in range(gy):
            diff = grid[x, y] - mean_val
            var_val += diff * diff
    var_val /= count
    if mean_val < 1e-12:
        return 0.0
    return var_val / (mean_val ** 2)

@njit
def basic_geometry(grid):
    """
    Return (width, height, varx, vary) from the grid distribution.
    """
    gx, gy = grid.shape
    total = 0.0
    sumx = 0.0
    sumy = 0.0
    minx, maxx = gx, 0
    miny, maxy = gy, 0

    for x in range(gx):
        for y in range(gy):
            w = grid[x, y]
            total += w
            sumx += x * w
            sumy += y * w
            if w > 0:
                if x < minx:
                    minx = x
                if x > maxx:
                    maxx = x
                if y < miny:
                    miny = y
                if y > maxy:
                    maxy = y

    if total < 1e-9:
        return 0.0, 0.0, 0.0, 0.0

    meanx = sumx / total
    meany = sumy / total

    varx, vary = 0.0, 0.0
    for x in range(gx):
        for y in range(gy):
            w = grid[x, y]
            if w > 0:
                dx = x - meanx
                dy = y - meany
                varx += dx * dx * w
                vary += dy * dy * w
    varx /= total
    vary /= total

    width = (maxx - minx + 1e-9) / gx
    height = (maxy - miny + 1e-9) / gy
    return width, height, varx, vary


def extract_fractal_features(params, iteration_depth=1):
    """
    Each fractal has 3 transformations => 18 parameters total.
    We do 1 iteration_depth call here (though you can adapt if you like).
    """
    transformations = np.array(params).reshape(-1, 6)  # shape = (3, 6)
    points = generate_fractal_points(transformations, NUM_FRAC_POINTS, seed=SEED)

    if len(points) < 2:
        return np.zeros(6, dtype=np.float64)

    grid = to_grid(points, GRID_SIZE)
    fd = fractal_dimension(grid, BOX_SIZES)
    lac = lacunarity(grid)
    w, h, vx, vy = basic_geometry(grid)
    return np.array([fd, lac, w, h, vx, vy], dtype=np.float64)

def adaptive_fractal_features(params, sample, max_depth=MAX_ITER_DEPTH):
    """
    A simple "adaptive" approach:
      - Start at depth=1
      - If fractal dimension < 1.5 => increment depth
        (up to max_depth) and re-generate features
    """
    depth_used = 1
    feats = extract_fractal_features(params, iteration_depth=depth_used)
    while depth_used < max_depth:
        if feats[0] < 1.5:
            depth_used += 1
            feats = extract_fractal_features(params, iteration_depth=depth_used)
        else:
            break
    return feats


# -----------------------------
# Evolutionary Search
# -----------------------------

def random_fractal():
    """
    We have 3 transformations => 18 params.
    Each transformation is [a, b, e, c, d, f].
    """
    frac = np.random.uniform(-1.0, 1.0, size=18) * PARAM_LIMIT
    return clamp_params(frac)

def mutate_fractal(frac):
    """Small random mutation in one param."""
    frac_new = frac.copy()
    idx = np.random.randint(len(frac_new))
    frac_new[idx] += np.random.normal(0, 1.0)
    return clamp_params(frac_new)

def crossover_fractal(frac1, frac2):
    """One-point crossover for demonstration."""
    pt = np.random.randint(1, len(frac1))
    child1 = np.concatenate([frac1[:pt], frac2[pt:]])
    child2 = np.concatenate([frac2[:pt], frac1[pt:]])
    return clamp_params(child1), clamp_params(child2)

def evaluate_fractal(frac, X_cluster, y_cluster, sample_size=30):
    """
    Evaluate fractal by:
      1) generating fractal features for a random subset
      2) training a small classifier (random forest with few estimators)
      3) returning balanced accuracy
    """
    if len(X_cluster) < 2:
        return 0.0

    sub_size = min(sample_size, len(X_cluster))
    idx = np.random.choice(len(X_cluster), sub_size, replace=False)
    X_sub = X_cluster[idx]
    y_sub = y_cluster[idx]

    from sklearn.ensemble import RandomForestClassifier

    # Build fractal features for these subset samples
    feats_sub = []
    for _ in X_sub:
        feats_sub.append(extract_fractal_features(frac))
    feats_sub = np.array(feats_sub)

    # If there's only one class in y_sub, balanced accuracy is meaningless => 0.0
    if len(np.unique(y_sub)) < 2:
        return 0.0

    # We'll train a small random forest (to keep it quick)
    clf = RandomForestClassifier(random_state=SEED, n_estimators=10)
    clf.fit(feats_sub, y_sub)
    preds = clf.predict(feats_sub)
    return balanced_accuracy_score(y_sub, preds)

def evolve_fractal_for_cluster(X_cluster, y_cluster):
    """
    Genetic search for the best fractal in a single cluster.
    Using pop_size=POP_SIZE, generations=GENERATIONS.
    """
    population = [random_fractal() for _ in range(POP_SIZE)]
    fitnesses = [evaluate_fractal(frac, X_cluster, y_cluster) for frac in population]

    best_idx = np.argmax(fitnesses)
    best_frac = population[best_idx]
    best_fit = fitnesses[best_idx]

    for g in range(GENERATIONS):
        new_pop = []
        while len(new_pop) < POP_SIZE:
            # tournament selection
            cand_idx = np.random.choice(POP_SIZE, 2, replace=False)
            f1 = fitnesses[cand_idx[0]]
            f2 = fitnesses[cand_idx[1]]
            parent1 = population[cand_idx[0]] if f1 > f2 else population[cand_idx[1]]

            cand_idx2 = np.random.choice(POP_SIZE, 2, replace=False)
            f3 = fitnesses[cand_idx2[0]]
            f4 = fitnesses[cand_idx2[1]]
            parent2 = population[cand_idx2[0]] if f3 > f4 else population[cand_idx2[1]]

            # crossover
            c1, c2 = crossover_fractal(parent1, parent2)
            # mutation
            if np.random.rand() < 0.3:
                c1 = mutate_fractal(c1)
            if np.random.rand() < 0.3:
                c2 = mutate_fractal(c2)

            new_pop.append(c1)
            if len(new_pop) < POP_SIZE:
                new_pop.append(c2)

        population = new_pop
        fitnesses = [evaluate_fractal(frac, X_cluster, y_cluster) for frac in population]
        gen_best_idx = np.argmax(fitnesses)
        gen_best_frac = population[gen_best_idx]
        gen_best_fit = fitnesses[gen_best_idx]
        if gen_best_fit > best_fit:
            best_frac = gen_best_frac
            best_fit = gen_best_fit

    return best_frac, best_fit


# -----------------------------
# Mixture-of-Experts Build & Predict
# -----------------------------

def build_fractal_moe(X, y, n_clusters=N_CLUSTERS):
    """
    1) Use KMeans to cluster data.
    2) For each cluster, run fractal evolution.
    3) Train a local RandomForest on fractal features for that cluster.
    """
    start_time = time.time()

    print(f"Performing KMeans with n_clusters={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)
    cluster_labels = kmeans.fit_predict(X)

    # cluster distribution
    for c_id in range(n_clusters):
        c_count = np.sum(cluster_labels == c_id)
        print(f"  - Cluster {c_id} size = {c_count}")

    fractal_params_list = []
    local_classifiers = []

    from sklearn.ensemble import RandomForestClassifier

    for c_id in range(n_clusters):
        print(f"\n--- Evolving fractal for cluster {c_id} ---")
        c_idx = np.where(cluster_labels == c_id)[0]
        Xc = X[c_idx]
        yc = y[c_idx]

        # trivial cluster check
        if len(Xc) < 2 or len(np.unique(yc)) < 2:
            best_frac = random_fractal()
            fractal_params_list.append(best_frac)
            local_classifiers.append(None)
            print(f"Cluster {c_id} => trivial cluster => random fractal, no local classifier.")
            continue

        # evolve fractal
        best_frac, best_fit = evolve_fractal_for_cluster(Xc, yc)
        fractal_params_list.append(best_frac)
        print(f"Cluster {c_id}: best fractal fitness after evolution = {best_fit:.4f}")

        # now train local classifier on entire cluster
        feats_c = []
        for _ in Xc:
            feats_c.append(extract_fractal_features(best_frac))
        feats_c = np.array(feats_c)

        unique_cls = np.unique(yc)
        if len(unique_cls) < 2:
            local_classifiers.append(None)
            print("  => Only one class => skip classifier.")
        else:
            clf = RandomForestClassifier(random_state=SEED, n_estimators=50)
            clf.fit(feats_c, yc)
            local_classifiers.append(clf)

            # measure cluster-level training score
            preds_c = clf.predict(feats_c)
            cluster_bal_acc = balanced_accuracy_score(yc, preds_c)
            print(f"  => Local classifier cluster-balanced-acc = {cluster_bal_acc:.4f}")

    end_time = time.time()
    train_time = end_time - start_time
    print(f"\nTraining completed in {train_time:.4f} seconds.")

    model = {
        'kmeans': kmeans,
        'fractal_params_list': fractal_params_list,
        'local_classifiers': local_classifiers,
        'train_time': train_time
    }
    return model

def predict_fractal_moe(model, X):
    """
    1) kmeans.predict -> cluster label
    2) fractal -> adaptive features
    3) local classifier => prediction
    """
    start_time = time.time()

    kmeans = model['kmeans']
    fractal_params_list = model['fractal_params_list']
    local_classifiers = model['local_classifiers']

    cluster_labels = kmeans.predict(X)
    preds = []

    for i, x in enumerate(X):
        c_id = cluster_labels[i]
        frac = fractal_params_list[c_id]
        clf = local_classifiers[c_id]

        feats = adaptive_fractal_features(frac, x, max_depth=MAX_ITER_DEPTH)

        if clf is None:
            pred_label = 0  # fallback if cluster had no classifier
        else:
            pred_label = clf.predict([feats])[0]

        preds.append(pred_label)

        if VERBOSE_PREDICT:
            print(f"Test sample {i}: cluster={c_id}, feats={feats}, pred={pred_label}")

    end_time = time.time()
    pred_time = end_time - start_time
    return np.array(preds), pred_time


# -----------------------------
# Main Demo
# -----------------------------

if __name__ == "__main__":
    print("\n===== Loading Breast Cancer Dataset =====\n")
    data = load_breast_cancer()
    X_data = data.data
    y_data = data.target
    # y_data: 0 = malignant, 1 = benign

    print(f"Dataset size: {X_data.shape}, #labels={len(y_data)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.3, random_state=SEED, stratify=y_data
    )

    print("\n===== Building Fractal Mixture-of-Experts Model =====\n")
    model = build_fractal_moe(X_train, y_train, n_clusters=N_CLUSTERS)

    print("\n===== Predicting on Test Set =====\n")
    y_pred, pred_time = predict_fractal_moe(model, X_test)

    from sklearn.metrics import balanced_accuracy_score
    acc = balanced_accuracy_score(y_test, y_pred)

    train_time = model['train_time']

    print("\n===== Final Results =====")
    print(f"Training Time   = {train_time:.4f} seconds")
    print(f"Prediction Time = {pred_time:.4f} seconds")
    print(f"Balanced Accuracy on Test Set = {acc:.4f}")
    print("============================================\n")
