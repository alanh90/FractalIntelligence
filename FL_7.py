"""
=== Building Hybrid Fractal Mixture-of-Experts ===

Cluster 0 => size=287
  Best fractal fitness: 0.5000
  Local cluster training balanced-acc: 0.5000
Cluster 1 => size=31
Cluster 2 => size=80
  Best fractal fitness: 0.5000
  Local cluster training balanced-acc: 0.5000

Hybrid MoE training completed in 2.3328 seconds.

=== Predicting on Test Set ===

Balanced Accuracy on Test Set = 0.8454
Training Time = 2.3328 s, Prediction Time = 0.2026 s

"""

import numpy as np
import math
import random
import time
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from numba import njit
from scipy.stats import moment

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Mixture-of-Experts
N_CLUSTERS = 3

# Evolution parameters
POP_SIZE = 10
GENERATIONS = 5

# For fractal param bounding
PARAM_LIMIT = 6.0

# Fractal generation
NUM_FRAC_POINTS = 2000  # bigger than before for better fractal detail
GRID_SIZE = 40
BOX_SIZES = np.array([2.0, 4.0, 8.0], dtype=np.float64)

# We'll store a base fractal for each cluster, plus a "refined" fractal
# (like the older code's approach).
NUM_TRANSFORMATIONS = 3  # how many transformations in each fractal
VERBOSE_PREDICT = False


# -----------------------------
# Numba-accelerated fractal generation & basic geometry
# -----------------------------

@njit
def clamp_params(params, limit):
    return np.clip(params, -limit, limit)


@njit
def generate_fractal_points(transformations, num_points, seed=0):
    """
    transformations shape = (num_transforms, 6)
    Each row => [a, b, e, c, d, f]
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

    # slope => fractal dimension
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
    varx = 0.0
    vary = 0.0
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


def extract_fractal_features(points):
    """
    Given fractal points, produce 6 features: [fd, lac, width, height, varx, vary].
    """
    if len(points) < 2:
        return np.zeros(6, dtype=np.float64)
    grid = to_grid(points, GRID_SIZE)
    fd = fractal_dimension(grid, BOX_SIZES)
    lac = lacunarity(grid)
    w, h, vx, vy = basic_geometry(grid)
    return np.array([fd, lac, w, h, vx, vy], dtype=np.float64)


# -----------------------------
# "Moment-based" parameter initialization
# -----------------------------

def create_moment_transformations(X_cluster):
    """
    Using old FastFractalFeatures logic:
      m1 = mean, m2 = std, m3 = 3rd moment, m4 = 4th moment
      Then define 3 transformations via some tanh-based formula.
    """
    m1 = np.mean(X_cluster)
    m2 = np.std(X_cluster) + 1e-9
    m3 = moment(X_cluster, moment=3, axis=None)
    m4 = moment(X_cluster, moment=4, axis=None)

    def single_transform(m1, m2, m3, m4):
        a = 0.5 * np.tanh(m1 + 0.2 * m2)
        b = 0.3 * np.tanh(m3)
        c = 0.3 * np.tanh(m4)
        d = 0.5 * np.tanh(m1 - 0.2 * m2)
        e = 0.1 * m1
        f = 0.1 * m2
        return np.array([a, b, e, c, d, f], dtype=np.float64)

    transforms = []
    for _ in range(NUM_TRANSFORMATIONS):
        row = single_transform(m1, m2, m3, m4)
        transforms.append(row)

    transforms = np.array(transforms)
    transforms = clamp_params(transforms, PARAM_LIMIT)
    return transforms


# -----------------------------
# Evolutionary code to refine cluster fractal
# -----------------------------

def random_transformations():
    """
    Random transformations: shape (NUM_TRANSFORMATIONS, 6)
    """
    t = np.random.uniform(-1.0, 1.0, size=(NUM_TRANSFORMATIONS, 6)) * PARAM_LIMIT
    return t


def mutate_transformations(T):
    T_new = T.copy()
    nrows, ncols = T_new.shape
    row = np.random.randint(nrows)
    col = np.random.randint(ncols)
    T_new[row, col] += np.random.normal(0, 1.0)
    return clamp_params(T_new, PARAM_LIMIT)


def crossover_transformations(T1, T2):
    nrows, ncols = T1.shape
    # pick row or column boundary as crossover?
    # here we do a single row-based crossover
    row_pt = np.random.randint(nrows)
    child1 = np.vstack([
        T1[:row_pt],
        T2[row_pt:]
    ])
    child2 = np.vstack([
        T2[:row_pt],
        T1[row_pt:]
    ])
    return clamp_params(child1, PARAM_LIMIT), clamp_params(child2, PARAM_LIMIT)


def evaluate_transformations(T, Xc, yc, sample_size=20):
    """
    Quick evaluation:
      1) Generate fractal points (2000 is big, so let's do a partial approach)
      2) For random subset of cluster => scale fractal => extract geometry => small classifier => balanced acc
    """
    # trivial cluster check
    if len(Xc) < 2:
        return 0.0

    sub_size = min(sample_size, len(Xc))
    idx = np.random.choice(len(Xc), sub_size, replace=False)
    X_sub = Xc[idx]
    y_sub = yc[idx]

    # We'll do a quick geometric feature extraction
    # Strategy: generate a "base fractal" once, then for each sample do simple scaling
    base_points = generate_fractal_points(T, NUM_FRAC_POINTS, seed=SEED)
    if len(base_points) < 2:
        return 0.0

    feats_sub = []
    for x in X_sub:
        sample_mean = np.mean(x)
        sample_std = np.std(x) + 1e-9
        scaled_pts = base_points * sample_std + sample_mean
        feats_sub.append(extract_fractal_features(scaled_pts))
    feats_sub = np.array(feats_sub)

    if len(np.unique(y_sub)) < 2:
        return 0.0

    clf = RandomForestClassifier(n_estimators=10, random_state=SEED)
    clf.fit(feats_sub, y_sub)
    preds = clf.predict(feats_sub)
    return balanced_accuracy_score(y_sub, preds)


def evolve_cluster_fractal(Xc, yc, init_T):
    """
    Evolve fractal transformations for cluster c, starting from moment-based init_T.
    """
    population = []
    fitnesses = []
    pop_size = POP_SIZE

    # seed half the population with random transforms, half with init_T plus small noise
    half = pop_size // 2
    for _ in range(half):
        # init + noise
        Tn = init_T.copy()
        Tn += np.random.normal(0, 0.1, Tn.shape)
        Tn = clamp_params(Tn, PARAM_LIMIT)
        population.append(Tn)
    for _ in range(pop_size - half):
        population.append(random_transformations())

    fitnesses = [evaluate_transformations(T, Xc, yc) for T in population]
    best_idx = np.argmax(fitnesses)
    best_T = population[best_idx]
    best_fit = fitnesses[best_idx]

    for g in range(GENERATIONS):
        new_pop = []
        new_fit = []
        while len(new_pop) < pop_size:
            cand_idx = np.random.choice(pop_size, 2, replace=False)
            p1 = population[cand_idx[0]]
            p2 = population[cand_idx[1]]
            f1 = fitnesses[cand_idx[0]]
            f2 = fitnesses[cand_idx[1]]
            if f1 > f2:
                parent1 = p1
            else:
                parent1 = p2

            cand_idx2 = np.random.choice(pop_size, 2, replace=False)
            p3 = population[cand_idx2[0]]
            p4 = population[cand_idx2[1]]
            f3 = fitnesses[cand_idx2[0]]
            f4 = fitnesses[cand_idx2[1]]
            if f3 > f4:
                parent2 = p3
            else:
                parent2 = p4

            c1, c2 = crossover_transformations(parent1, parent2)
            if np.random.rand() < 0.3:
                c1 = mutate_transformations(c1)
            if np.random.rand() < 0.3:
                c2 = mutate_transformations(c2)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop
        fitnesses = [evaluate_transformations(T, Xc, yc) for T in population]
        gen_best_idx = np.argmax(fitnesses)
        gen_best_T = population[gen_best_idx]
        gen_best_fit = fitnesses[gen_best_idx]
        if gen_best_fit > best_fit:
            best_T = gen_best_T
            best_fit = gen_best_fit

    return best_T, best_fit


# -----------------------------
# Building the Hybrid MoE
# -----------------------------

def build_hybrid_moe(X, y, n_clusters=N_CLUSTERS):
    """
    1) KMeans => cluster assignments
    2) For each cluster:
       a) compute moment transformations
       b) small evolution to refine
       c) pre-generate a "base fractal" using the refined transformations
       d) train local classifier
    """
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)
    cluster_labels = kmeans.fit_predict(X)

    fractal_params_list = []
    base_fractals_list = []
    local_classifiers = []

    for c_id in range(n_clusters):
        c_idx = np.where(cluster_labels == c_id)[0]
        Xc = X[c_idx]
        yc = y[c_idx]
        print(f"Cluster {c_id} => size={len(Xc)}")

        if len(Xc) < 2 or len(np.unique(yc)) < 2:
            # trivial
            fractal_params_list.append(None)
            base_fractals_list.append(None)
            local_classifiers.append(None)
            continue

        # 1) moment-based initial transformations
        init_T = create_moment_transformations(Xc)
        # 2) refine via evolution
        best_T, best_fit = evolve_cluster_fractal(Xc, yc, init_T)
        fractal_params_list.append(best_T)
        print(f"  Best fractal fitness: {best_fit:.4f}")

        # 3) Pre-generate a base fractal pattern
        #    We'll store it so we don't have to re-generate fractal points for each sample
        base_pts = generate_fractal_points(best_T, NUM_FRAC_POINTS, seed=SEED)
        base_fractals_list.append(base_pts)

        # 4) Build local classifier using "scaled" fractal features for each sample
        feats_c = []
        for x_smpl in Xc:
            smpl_mean = np.mean(x_smpl)
            smpl_std = np.std(x_smpl) + 1e-9
            scaled_pts = base_pts * smpl_std + smpl_mean
            feats_c.append(extract_fractal_features(scaled_pts))
        feats_c = np.array(feats_c)

        if len(np.unique(yc)) < 2:
            local_classifiers.append(None)
            continue
        clf = RandomForestClassifier(n_estimators=50, random_state=SEED)
        clf.fit(feats_c, yc)
        local_classifiers.append(clf)

        preds_c = clf.predict(feats_c)
        cluster_bal_acc = balanced_accuracy_score(yc, preds_c)
        print(f"  Local cluster training balanced-acc: {cluster_bal_acc:.4f}")

    train_time = time.time() - start_time
    print(f"\nHybrid MoE training completed in {train_time:.4f} seconds.")
    return {
        'kmeans': kmeans,
        'fractal_params_list': fractal_params_list,
        'base_fractals_list': base_fractals_list,
        'local_classifiers': local_classifiers,
        'train_time': train_time
    }


def predict_hybrid_moe(model, X):
    start_time = time.time()
    kmeans = model['kmeans']
    fractal_params_list = model['fractal_params_list']
    base_fractals_list = model['base_fractals_list']
    local_classifiers = model['local_classifiers']

    cluster_labels = kmeans.predict(X)
    preds = []

    for i, x in enumerate(X):
        c_id = cluster_labels[i]
        T = fractal_params_list[c_id]
        base_pts = base_fractals_list[c_id]
        clf = local_classifiers[c_id]
        if T is None or base_pts is None or clf is None:
            # trivial => default class
            preds.append(0)
            continue

        # do the "scaling" approach
        smpl_mean = np.mean(x)
        smpl_std = np.std(x) + 1e-9
        scaled_pts = base_pts * smpl_std + smpl_mean

        # optionally do small "adaptive" approach (like the old code's refined fractal)
        # but for brevity, let's skip a second fractal pass—though you could add it.
        feats = extract_fractal_features(scaled_pts)

        pred_label = clf.predict([feats])[0]
        preds.append(pred_label)

        if VERBOSE_PREDICT:
            print(f"Test sample {i} => cluster {c_id}, pred={pred_label}")

    pred_time = time.time() - start_time
    return np.array(preds), pred_time


# -----------------------------
# Demo on Breast Cancer
# -----------------------------

if __name__ == "__main__":
    data = load_breast_cancer()
    X_data = data.data
    y_data = data.target
    print(f"Breast Cancer dataset shape: {X_data.shape}, classes={np.unique(y_data)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.3, random_state=SEED, stratify=y_data
    )

    print("\n=== Building Hybrid Fractal Mixture-of-Experts ===\n")
    model = build_hybrid_moe(X_train, y_train, n_clusters=N_CLUSTERS)

    print("\n=== Predicting on Test Set ===\n")
    y_pred, pred_time = predict_hybrid_moe(model, X_test)
    acc = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced Accuracy on Test Set = {acc:.4f}")
    print(f"Training Time = {model['train_time']:.4f} s, Prediction Time = {pred_time:.4f} s")
