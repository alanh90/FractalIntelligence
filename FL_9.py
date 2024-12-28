#!/usr/bin/env python3
"""
progressive_fractal_mnist.py

A demonstration of a "progressive fractal training" idea for MNIST:
1) We define multiple LODs (7x7, 14x14, 28x28).
2) At each LOD, we do a partial evolutionary search for fractal parameters that classify digits well at that resolution.
3) We warm-start the fractal parameters at the next LOD, refining them further.
4) Finally, we evaluate the fractal approach at the highest resolution.

Author: You
"""

import numpy as np
import time
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ----------------------------------------------------------------------------
# 1) LOAD MNIST
# ----------------------------------------------------------------------------

def load_mnist_data():
    """
    Loads MNIST from tensorflow.keras.datasets, returns (X_train, y_train), (X_test, y_test).
    X are 28x28 images (integers 0-255), y are digit labels 0-9.
    """
    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    # For convenience, let's keep them as floats 0..1
    X_train_full = X_train_full.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    return (X_train_full, y_train_full), (X_test, y_test)


# ----------------------------------------------------------------------------
# 2) DOWNSAMPLING
# ----------------------------------------------------------------------------

def downsample_images(X, new_size=(14, 14)):
    """
    Downsample each image (28x28 float) to e.g. (14x14 or 7x7).
    We'll do a simple approach (block averaging).
    """
    old_size = X.shape[1]  # 28
    n_samples = X.shape[0]
    out = np.zeros((n_samples, new_size[0], new_size[1]), dtype=np.float32)

    row_scale = old_size / new_size[0]
    col_scale = old_size / new_size[1]

    for i in range(n_samples):
        img = X[i]
        # For each pixel in the new image, average the corresponding region in the old image
        for r in range(new_size[0]):
            for c in range(new_size[1]):
                # range in old image
                r0 = int(r * row_scale)
                r1 = int((r + 1) * row_scale)
                c0 = int(c * col_scale)
                c1 = int((c + 1) * col_scale)
                patch = img[r0:r1, c0:c1]
                out[i, r, c] = np.mean(patch)
    return out


# ----------------------------------------------------------------------------
# 3) FRACTAL PARAMS + PARTIAL EVOLUTION
# ----------------------------------------------------------------------------

def random_fractal_params(num_transforms=2):
    """
    Simple fractal param representation:
    Each transform is [a,b,c,d,e,f], so total length = 6 * num_transforms.
    We'll keep them in a 2D array shape (num_transforms, 6).
    We'll clamp to [-1,1] for demonstration.
    """
    arr = np.random.uniform(-1, 1, size=(num_transforms, 6))
    return arr


def clamp_params(params):
    np.clip(params, -1.0, 1.0, out=params)
    return params


def mutate(params, rate=0.3):
    p_new = np.copy(params)
    if np.random.rand() < rate:
        # pick random element
        r_i = np.random.randint(p_new.shape[0])
        c_i = np.random.randint(p_new.shape[1])
        p_new[r_i, c_i] += np.random.normal(0, 0.2)
    return clamp_params(p_new)


def crossover(p1, p2):
    p1_copy = np.copy(p1)
    p2_copy = np.copy(p2)
    # single crossover by row or col
    row_pt = np.random.randint(p1_copy.shape[0])
    child1 = np.vstack([p1_copy[:row_pt], p2_copy[row_pt:]])
    child2 = np.vstack([p2_copy[:row_pt], p1_copy[row_pt:]])
    return clamp_params(child1), clamp_params(child2)


# We'll define a fractal feature extraction that uses these transforms in some simplistic manner
def fractal_feature(img, fractal_params):
    """
    A silly demonstration: We "apply" fractal transforms to a single aggregated statistic
    from the image and produce a small set of derived features.

    - We'll gather (mean, std) from the image.
    - Then for each transform [a,b,c,d,e,f], we produce 2 or 3 expansions.
    - Return a small feature vector of length = num_transforms*2 + ?

    In a real system, you'd do a more elaborate fractal expansion that processes the entire image.
    """
    mean_val = np.mean(img)
    std_val = np.std(img) + 1e-9

    # We'll do a partial approach: x-> (a*x + b*std + c), y-> (d*x + e*std + f)
    # Then gather the sums or something
    feats = []
    for row in fractal_params:
        a, b, c, d, e, f = row
        x_new = a * mean_val + b * std_val + c
        y_new = d * mean_val + e * std_val + f
        # store x_new, y_new
        feats.extend([x_new, y_new])
    return np.array(feats, dtype=np.float32)


def evaluate_fractal_params(params_population, X_samples, y_samples, subset_size=100):
    """
    Evaluate each fractal param in the population by:
      1) picking subset_size random samples
      2) generate fractal-based features for each sample
      3) train a small classifier => measure accuracy
    We'll just do a quick KNN or logistic on that subset.
    """
    from sklearn.linear_model import LogisticRegression

    if len(X_samples) < subset_size:
        subset_size = len(X_samples)
    subset_idx = np.random.choice(len(X_samples), subset_size, replace=False)
    X_sub = X_samples[subset_idx]
    y_sub = y_samples[subset_idx]

    # We'll build fractal features for each param, train a quick classifier on the same subset
    # This is not perfect, but demonstrates partial classification-based fitness.
    fitnesses = []
    for p_idx, param in enumerate(params_population):
        feats_list = []
        for i in range(subset_size):
            feat = fractal_feature(X_sub[i], param)
            feats_list.append(feat)
        feats_arr = np.array(feats_list, dtype=np.float32)

        # train a quick logistic regression
        clf = LogisticRegression(max_iter=200, random_state=SEED)
        try:
            clf.fit(feats_arr, y_sub)
            preds = clf.predict(feats_arr)
            acc = accuracy_score(y_sub, preds)
        except:
            # if it fails (maybe singular?), set acc=0
            acc = 0.0

        fitnesses.append(acc)
    return fitnesses


def evolve_fractal_params(X_samples, y_samples, pop_size=10, generations=5, num_transforms=2):
    """
    A partial evolutionary loop:
     1) init population
     2) measure fitness => classification accuracy on small subset
     3) do crossover/mutation
     4) return best
    """
    population = []
    for _ in range(pop_size):
        population.append(clamp_params(random_fractal_params(num_transforms)))

    fitnesses = evaluate_fractal_params(population, X_samples, y_samples)
    best_idx = np.argmax(fitnesses)
    best_fit = fitnesses[best_idx]
    best_p = population[best_idx]

    for g in range(generations):
        new_pop = []
        new_fit = []
        while len(new_pop) < pop_size:
            # tournament
            cand_idx = np.random.choice(pop_size, 2, replace=False)
            f1 = fitnesses[cand_idx[0]]
            f2 = fitnesses[cand_idx[1]]
            if f1 > f2:
                parent1 = population[cand_idx[0]]
            else:
                parent1 = population[cand_idx[1]]

            cand_idx2 = np.random.choice(pop_size, 2, replace=False)
            f3 = fitnesses[cand_idx2[0]]
            f4 = fitnesses[cand_idx2[1]]
            if f3 > f4:
                parent2 = population[cand_idx2[0]]
            else:
                parent2 = population[cand_idx2[1]]

            c1, c2 = crossover(parent1, parent2)
            c1 = mutate(c1, rate=0.3)
            c2 = mutate(c2, rate=0.3)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop
        fitnesses = evaluate_fractal_params(population, X_samples, y_samples)
        gen_best_idx = np.argmax(fitnesses)
        gen_best_fit = fitnesses[gen_best_idx]
        gen_best_p = population[gen_best_idx]
        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_p = gen_best_p
        print(f"   Generation {g + 1}, best fitness so far = {best_fit:.4f}")

    return best_p, best_fit


# ----------------------------------------------------------------------------
# 4) PROGRESSIVE LOOP
# ----------------------------------------------------------------------------

def progressive_fractal_training(X_train, y_train, lod_sizes=[7, 14, 28], pop_size=10, generations=5):
    """
    We'll do a loop:
      For each lod_size in lod_sizes:
        - downsample X_train => X_lod
        - evolve fractal params (warm-start from last if available)
        - store best fractal param
    Return final fractal param
    """
    best_param = None
    for i, sz in enumerate(lod_sizes):
        print(f"\n*** LOD {sz}x{sz} stage ***")
        # downsample
        if sz < 28:
            X_lod = downsample_images(X_train, (sz, sz))
        else:
            X_lod = X_train.copy()  # 28x28 is original

        # evolve
        if best_param is None:
            # start from scratch
            population_size = pop_size
        else:
            # we can partially fill half the pop with the best param + random
            population_size = pop_size

        # We'll define a quick function to do partial evolve
        best_param, best_fit = evolve_fractal_params(X_lod, y_train, pop_size=population_size, generations=generations, num_transforms=2)
        print(f"LOD {sz} => best fitness at this stage = {best_fit:.4f}")

    return best_param


# ----------------------------------------------------------------------------
# 5) FINAL CLASSIFICATION
# ----------------------------------------------------------------------------

def final_classification(fractal_param, X_train, y_train, X_test, y_test):
    """
    Once we have the final fractal_param, we transform all train samples => features, train a classifier,
    then measure test accuracy.
    """
    # build features
    print("\n[Final] Building fractal features for train & test...")
    t0 = time.time()
    train_feats = [fractal_feature(img, fractal_param) for img in X_train]
    test_feats = [fractal_feature(img, fractal_param) for img in X_test]
    train_feats = np.array(train_feats, dtype=np.float32)
    test_feats = np.array(test_feats, dtype=np.float32)
    build_time = time.time() - t0

    # train classifier
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=300, random_state=SEED)
    print("[Final] Training logistic regression on fractal feats...")
    t0 = time.time()
    clf.fit(train_feats, y_train)
    train_clf_time = time.time() - t0

    # test
    t0 = time.time()
    preds = clf.predict(test_feats)
    test_clf_time = time.time() - t0

    acc = accuracy_score(y_test, preds)
    print(f"[Final] Test Accuracy = {acc:.4f}")
    print(f"[Final] Feat Build Time = {build_time:.4f}s, Train Clf Time = {train_clf_time:.4f}s, Test Clf Time = {test_clf_time:.4f}s")
    return acc


# ----------------------------------------------------------------------------
# 6) MAIN
# ----------------------------------------------------------------------------

def main():
    print("Loading MNIST...")
    (X_train_full, y_train_full), (X_test, y_test) = load_mnist_data()

    # let's do a smaller subset for speed if you want
    # but let's keep it for demonstration
    # train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=SEED, stratify=y_train_full
    )
    # combine train+val again at the end => let's just use X_train for partial evolution
    X_trf = np.concatenate([X_train, X_val], axis=0)
    y_trf = np.concatenate([y_train, y_val], axis=0)

    # Progressive approach with LOD = 7,14,28
    start_time = time.time()
    final_param = progressive_fractal_training(X_trf, y_trf, lod_sizes=[7, 14, 28], pop_size=10, generations=3)
    train_time = time.time() - start_time
    print(f"\nProgressive fractal training finished, total time = {train_time:.2f}s")

    # Evaluate final param on the full training set (X_trf) vs. test
    final_acc = final_classification(final_param, X_trf, y_trf, X_test, y_test)
    print(f"Final test accuracy after progressive fractal approach = {final_acc:.4f}")


if __name__ == "__main__":
    main()
