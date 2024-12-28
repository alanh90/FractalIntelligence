"""
hybrid_class_fractals_fixed.py

A revised version of the hybrid approach that:
1) Creates a fractal per class using class-level moments (like ClassLevelFractals).
2) Partially evolves those fractals with an extended population/generations.
3) Stores base + refined fractals for each class.
4) At inference, picks the nearest class centroid's fractal, scales it by the sample's mean/std,
   and extracts bounding-box geometry features (20 features total).
5) Uses KNN for final classification.

Fixes / Improvements:
- POP_SIZE=20, GENERATIONS=5, sample_size=50 for more robust evolution.
- Extra logging: prints distribution of y_train, logs partial evolution progress, etc.
- Avoid skipping classes silently; logs a message if a class is truly trivial.
"""

import numpy as np
import math
import random
import time
from scipy.stats import moment
from numba import njit
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Updated evolutionary parameters
POP_SIZE = 20
GENERATIONS = 5
PARAM_LIMIT = 6.0

# For fractal generation
NUM_TRANSFORMATIONS = 3   # 3 transformations per fractal
NUM_POINTS = 2000         # number of points to generate per fractal

# We'll increase the sample_size used in fractal fitness to 50
FITNESS_SAMPLE_SIZE = 50

# Logging verbosity
VERBOSE_EVOLUTION = True  # if True, prints debug info each generation


@njit
def clamp_params(params, limit):
    return np.clip(params, -limit, limit)


@njit
def generate_fractal_points(transformations, num_points, seed=0):
    """
    transformations shape = (num_transforms, 6)
    Each row => [a, b, e, c, d, f].
    We'll generate up to 'num_points' points in typical IFS style.
    """
    np.random.seed(seed)
    x, y = 0.0, 0.0
    t_count = transformations.shape[0]
    points = np.empty((num_points, 2), dtype=np.float64)
    idx_count = 0

    for i in range(num_points):
        idx = np.random.randint(t_count)
        t = transformations[idx]
        x_new = t[0]*x + t[1]*y + t[2]
        y_new = t[3]*x + t[4]*y + t[5]
        if not (math.isfinite(x_new) and math.isfinite(y_new)):
            break
        if abs(x_new) > 1e6 or abs(y_new) > 1e6:
            break
        x, y = x_new, y_new
        points[idx_count, 0] = x
        points[idx_count, 1] = y
        idx_count += 1

    return points[:idx_count]


def random_transformations():
    """
    Build random transformations for a fractal with NUM_TRANSFORMATIONS rows,
    each row is [a, b, e, c, d, f].
    """
    T = np.random.uniform(-1, 1, size=(NUM_TRANSFORMATIONS, 6)) * PARAM_LIMIT
    return T


def mutate_transformations(T):
    T_new = T.copy()
    nrows, ncols = T_new.shape
    row = np.random.randint(nrows)
    col = np.random.randint(ncols)
    T_new[row, col] += np.random.normal(0, 1.0)
    return clamp_params(T_new, PARAM_LIMIT)


def crossover_transformations(T1, T2):
    nrows, ncols = T1.shape
    pt = np.random.randint(1, nrows)  # row-based crossover
    child1 = np.vstack([T1[:pt], T2[pt:]])
    child2 = np.vstack([T2[:pt], T1[pt:]])
    child1 = clamp_params(child1, PARAM_LIMIT)
    child2 = clamp_params(child2, PARAM_LIMIT)
    return child1, child2


def extract_geometric_features(points):
    """
    Simple bounding-box style geometry:
    width, height, x_mean, y_mean, x_var, y_var, dist_min, dist_max, dist_mean, point_count
    => 10 features
    """
    if len(points) == 0:
        return np.zeros(10, dtype=np.float64)

    x_coords = points[:, 0]
    y_coords = points[:, 1]

    xmin, xmax = np.min(x_coords), np.max(x_coords)
    ymin, ymax = np.min(y_coords), np.max(y_coords)
    width = xmax - xmin
    height = ymax - ymin

    xm = np.mean(x_coords)
    ym = np.mean(y_coords)
    xv = np.var(x_coords)
    yv = np.var(y_coords)

    dists = np.sqrt((x_coords - xm)**2 + (y_coords - ym)**2)
    dmin = np.min(dists)
    dmax = np.max(dists)
    dmean = np.mean(dists)

    return np.array([width, height, xm, ym, xv, yv, dmin, dmax, dmean, len(points)], dtype=np.float64)


def evaluate_fractal(T, X_class, y_class, sample_size=FITNESS_SAMPLE_SIZE):
    """
    Evaluate fractal by generating fractal points once (NUM_POINTS),
    then for 'sample_size' random samples from X_class =>
    scale fractal => geometry => train small classifier => measure balanced acc
    """
    if len(X_class) < 2:
        return 0.0

    base_pts = generate_fractal_points(T, NUM_POINTS, seed=SEED)
    if len(base_pts) < 2:
        return 0.0

    sub_size = min(sample_size, len(X_class))
    idx = np.random.choice(len(X_class), sub_size, replace=False)
    X_sub = X_class[idx]
    y_sub = y_class[idx]

    # skip if only one label in the subset
    if len(np.unique(y_sub)) < 2:
        return 0.0

    feats_sub = []
    for sample in X_sub:
        sm = np.mean(sample)
        ss = np.std(sample) + 1e-9
        # scale the base fractal
        scaled_pts = base_pts * ss + sm
        feats_sub.append(extract_geometric_features(scaled_pts))
    feats_sub = np.array(feats_sub)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import balanced_accuracy_score

    # small classifier for fitness
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(feats_sub, y_sub)
    preds = clf.predict(feats_sub)
    return balanced_accuracy_score(y_sub, preds)


def moment_init_transformations(X_class):
    """
    Like the old 'ClassLevelFractals' approach: get (m1,m2,m3,m4) => build transformations for a single class.
    """
    from scipy.stats import moment
    m1 = np.mean(X_class)
    m2 = np.std(X_class) + 1e-9
    m3 = moment(X_class, moment=3, axis=None)
    m4 = moment(X_class, moment=4, axis=None)

    def single_transform(m1, m2, m3, m4):
        a = 0.5*np.tanh(m1 + 0.2*m2)
        b = 0.3*np.tanh(m3)
        c = 0.3*np.tanh(m4)
        d = 0.5*np.tanh(m1 - 0.2*m2)
        e = 0.1*m1
        f = 0.1*m2
        return np.array([a,b,e,c,d,f], dtype=np.float64)

    transforms = []
    for _ in range(NUM_TRANSFORMATIONS):
        row = single_transform(m1, m2, m3, m4)
        transforms.append(row)
    transforms = np.array(transforms)
    transforms = clamp_params(transforms, PARAM_LIMIT)
    return transforms


def partial_evolve_class_fractal(X_class, y_class, init_T):
    """
    Evolve fractal transformations for a single class.
    Using pop_size=20, generations=5, and logs debug info if VERBOSE_EVOLUTION=True.
    """
    population = []
    half = POP_SIZE // 2

    # seed half from init_T + noise, half random
    for _ in range(half):
        Tn = init_T.copy()
        Tn += np.random.normal(0, 0.1, Tn.shape)
        Tn = clamp_params(Tn, PARAM_LIMIT)
        population.append(Tn)
    for _ in range(POP_SIZE - half):
        population.append(random_transformations())

    fitnesses = [evaluate_fractal(T, X_class, y_class) for T in population]
    best_idx = np.argmax(fitnesses)
    best_T = population[best_idx]
    best_fit = fitnesses[best_idx]

    if VERBOSE_EVOLUTION:
        print(f"  [Init] best fitness = {best_fit:.4f}")

    for g in range(GENERATIONS):
        new_pop = []
        new_fit = []
        while len(new_pop) < POP_SIZE:
            cand_idx = np.random.choice(POP_SIZE, 2, replace=False)
            f1 = fitnesses[cand_idx[0]]
            f2 = fitnesses[cand_idx[1]]
            if f1 > f2:
                parent1 = population[cand_idx[0]]
            else:
                parent1 = population[cand_idx[1]]

            cand_idx2 = np.random.choice(POP_SIZE, 2, replace=False)
            f3 = fitnesses[cand_idx2[0]]
            f4 = fitnesses[cand_idx2[1]]
            if f3 > f4:
                parent2 = population[cand_idx2[0]]
            else:
                parent2 = population[cand_idx2[1]]

            c1, c2 = crossover_transformations(parent1, parent2)
            if np.random.rand() < 0.3:
                c1 = mutate_transformations(c1)
            if np.random.rand() < 0.3:
                c2 = mutate_transformations(c2)

            new_pop.append(c1)
            if len(new_pop) < POP_SIZE:
                new_pop.append(c2)

        population = new_pop
        fitnesses = [evaluate_fractal(T, X_class, y_class) for T in population]
        gen_best_idx = np.argmax(fitnesses)
        gen_best_T = population[gen_best_idx]
        gen_best_fit = fitnesses[gen_best_idx]
        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_T = gen_best_T

        if VERBOSE_EVOLUTION:
            print(f"  [Gen {g+1}] best fitness = {best_fit:.4f}")

    return best_T, best_fit


class HybridClassFractals:
    """
    For each class:
      - compute class centroid
      - moment-based init of fractal transformations
      - partial evolution
      - store best transformations
      - pre-generate base fractal + refined fractal
    At inference:
      - find nearest class centroid
      - scale/shift base+refined fractals with sample's mean/std => 20 features
    """
    def __init__(self):
        self.classes_ = None
        self.class_centroids = {}
        self.class_transformations = {}
        self.class_base_fractals = {}
        self.class_refined_fractals = {}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            idx = np.where(y == c)[0]
            Xc = X[idx]
            print(f"  Class {c}: {len(Xc)} samples, label dist: {np.unique(y[idx], return_counts=True)}")

            # If not enough samples, skip
            if len(Xc) < 2:
                print(f"  => Class {c} is trivial (fewer than 2 samples).")
                self.class_centroids[c] = np.zeros(X.shape[1])
                self.class_transformations[c] = None
                self.class_base_fractals[c] = None
                self.class_refined_fractals[c] = None
                continue

            # compute centroid
            centroid = np.mean(Xc, axis=0)
            self.class_centroids[c] = centroid

            # if truly single-label, skip
            if len(np.unique(y[idx])) < 1:
                print(f"  => Class {c} has 0 or 1 unique label?? Shouldn't happen. Skipping.")
                self.class_transformations[c] = None
                self.class_base_fractals[c] = None
                self.class_refined_fractals[c] = None
                continue

            # moment-based init
            init_T = moment_init_transformations(Xc)
            # partial evolve
            best_T, best_fit = partial_evolve_class_fractal(Xc, y[idx], init_T)
            print(f"  => Final best fractal fitness for class {c}: {best_fit:.4f}")
            self.class_transformations[c] = best_T

            # generate base fractal
            base_pts = generate_fractal_points(best_T, NUM_POINTS, seed=SEED)
            self.class_base_fractals[c] = base_pts

            # refined fractal
            refined_T = best_T.copy() + np.random.normal(0, 0.01, best_T.shape)
            refined_pts = generate_fractal_points(refined_T, NUM_POINTS, seed=SEED+1)
            self.class_refined_fractals[c] = refined_pts

    def _nearest_class(self, sample):
        # Euclidean distance to each class centroid
        best_c = None
        best_d = 1e15
        for c in self.classes_:
            dist = np.linalg.norm(sample - self.class_centroids[c])
            if dist < best_d:
                best_d = dist
                best_c = c
        return best_c

    def transform(self, sample):
        if self.classes_ is None or len(self.classes_) == 0:
            return np.zeros(20, dtype=np.float64)

        c = self._nearest_class(sample)
        base_pts = self.class_base_fractals[c]
        refined_pts = self.class_refined_fractals[c]
        if base_pts is None or refined_pts is None:
            # trivial class => fallback
            return np.zeros(20, dtype=np.float64)

        sm = np.mean(sample)
        ss = np.std(sample) + 1e-9

        base_scaled = base_pts * ss + sm
        refined_scaled = refined_pts * (ss+0.1) + (sm-0.05)

        feats_base = extract_geometric_features(base_scaled)
        feats_refined = extract_geometric_features(refined_scaled)
        return np.concatenate([feats_base, feats_refined])


def run_experiment_breast_cancer():
    print("===== Loading Breast Cancer Dataset =====")
    data = load_breast_cancer()
    X_data = data.data
    y_data = data.target
    print(f"Dataset shape: {X_data.shape}, classes={np.unique(y_data)}")

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.3, random_state=SEED, stratify=y_data
    )

    # Print distribution
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    print("Training class distribution:", dict(zip(unique_train, counts_train)))

    # optional scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Build fractals
    model = HybridClassFractals()

    start_time = time.time()
    model.fit(X_train, y_train)
    fractal_fit_time = time.time() - start_time
    print(f"\nFractal fit time = {fractal_fit_time:.4f}s")

    # Transform train
    start_time = time.time()
    X_train_fractal = np.array([model.transform(x) for x in X_train])
    X_test_fractal = np.array([model.transform(x) for x in X_test])
    transform_time = time.time() - start_time
    print(f"Transform time = {transform_time:.4f}s")

    # Train final KNN
    start_time = time.time()
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_fractal, y_train)
    classifier_train_time = time.time() - start_time
    print(f"Classifier train time = {classifier_train_time:.4f}s")

    # Predict
    start_time = time.time()
    preds = knn.predict(X_test_fractal)
    predict_time = time.time() - start_time
    print(f"Predict time = {predict_time:.4f}s")

    acc = balanced_accuracy_score(y_test, preds)
    print(f"Balanced Accuracy = {acc:.4f}")

    total_train_time = fractal_fit_time + transform_time + classifier_train_time
    print(f"Total training time = {total_train_time:.4f}s, Prediction time = {predict_time:.4f}s")


if __name__ == "__main__":
    run_experiment_breast_cancer()

'''
Output:

===== Loading Breast Cancer Dataset =====
Dataset shape: (569, 30), classes=[0 1]
Training class distribution: {np.int64(0): np.int64(148), np.int64(1): np.int64(250)}
  Class 0: 148 samples, label dist: (array([0]), array([148]))
  [Init] best fitness = 0.0000
  [Gen 1] best fitness = 0.0000
  [Gen 2] best fitness = 0.0000
  [Gen 3] best fitness = 0.0000
  [Gen 4] best fitness = 0.0000
  [Gen 5] best fitness = 0.0000
  => Final best fractal fitness for class 0: 0.0000
  Class 1: 250 samples, label dist: (array([1]), array([250]))
  [Init] best fitness = 0.0000
  [Gen 1] best fitness = 0.0000
  [Gen 2] best fitness = 0.0000
  [Gen 3] best fitness = 0.0000
  [Gen 4] best fitness = 0.0000
  [Gen 5] best fitness = 0.0000
  => Final best fractal fitness for class 1: 0.0000

Fractal fit time = 0.4786s
Transform time = 0.0654s
Classifier train time = 0.0010s
Predict time = 0.2189s
Balanced Accuracy = 0.9079
Total training time = 0.5449s, Prediction time = 0.2189s
'''
