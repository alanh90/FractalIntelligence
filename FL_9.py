#!/usr/bin/env python3
"""
fast_progressive_fractal_mnist.py

Goals:
 - Achieve ~85-90% on MNIST
 - Very fast training (small evolution, partial data)
 - Progressive approach: LOD14 -> LOD28
 - Fractal param approach with multiple transforms
 - Keep a 3D visualization of fractal evolution
 - Ensemble of top fractals for final classification

Author: You
"""

import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D scatter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from typing import List, Tuple

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================
# 1) LOADING & DOWNSAMPLING
# ============================

def load_mnist_data():
    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    # 0..1 floats
    X_train_full = X_train_full.astype(np.float32)/255.0
    X_test       = X_test.astype(np.float32)/255.0
    return (X_train_full, y_train_full), (X_test, y_test)

def downsample_images(X, new_size=(14,14)):
    """
    Quick average-pooling approach from 28x28 -> e.g. 14x14.
    """
    old_size = X.shape[1]  # 28 if not already downsampled
    n_samples = X.shape[0]
    out = np.zeros((n_samples, new_size[0], new_size[1]), dtype=np.float32)

    row_scale = old_size / new_size[0]
    col_scale = old_size / new_size[1]

    for i in range(n_samples):
        img = X[i]
        for r in range(new_size[0]):
            for c in range(new_size[1]):
                r0 = int(r*row_scale)
                r1 = int((r+1)*row_scale)
                c0 = int(c*col_scale)
                c1 = int((c+1)*col_scale)
                patch = img[r0:r1, c0:c1]
                out[i, r, c] = np.mean(patch)
    return out

# ============================
# 2) FRACTAL PARAMS & EVOLUTION
# ============================

def random_fractal_params(num_transforms=4):
    """
    Each transform is [a,b,c,d,e,f].
    shape => (num_transforms, 6).
    We'll clamp to [-1,1].
    """
    arr = np.random.uniform(-1,1,size=(num_transforms,6))
    return arr

def clamp_params(params):
    np.clip(params, -1.0,1.0,out=params)
    return params

def mutate(params, rate=0.3):
    p_new = np.copy(params)
    if np.random.rand()<rate:
        r_i = np.random.randint(p_new.shape[0])
        c_i = np.random.randint(p_new.shape[1])
        p_new[r_i, c_i] += np.random.normal(0,0.2)
    return clamp_params(p_new)

def crossover(p1, p2):
    # single row-based crossover
    row_pt = np.random.randint(p1.shape[0])
    c1 = np.vstack([p1[:row_pt], p2[row_pt:]])
    c2 = np.vstack([p2[:row_pt], p1[row_pt:]])
    return clamp_params(c1), clamp_params(c2)

def fractal_feature(img, fractal_params):
    """
    We'll do a simple approach again:
      - compute mean, std of the image
      - for each transform [a,b,c,d,e,f], produce x_new,y_new => final feats
    => total length = num_transforms*2
    """
    mean_val = np.mean(img)
    std_val  = np.std(img)+1e-9

    feats=[]
    for row in fractal_params:
        a,b,c,d,e,f = row
        x_new = a*mean_val + b*std_val + c
        y_new = d*mean_val + e*std_val + f
        feats.extend([x_new,y_new])
    return np.array(feats, dtype=np.float32)

def evaluate_fractal_population(
    population,
    X_samples,
    y_samples,
    subset_size=500,
):
    """
    Evaluate classification accuracy on a random subset.
    We'll do a small logistic regression as the classifier.
    Return a list of fitnesses.
    """
    if len(X_samples)<subset_size:
        subset_size=len(X_samples)
    idx_sub = np.random.choice(len(X_samples), subset_size, replace=False)
    X_sub = X_samples[idx_sub]
    y_sub = y_samples[idx_sub]

    from sklearn.linear_model import LogisticRegression

    fitnesses = []
    for param in population:
        feats_list = []
        for i in range(subset_size):
            feats_list.append(fractal_feature(X_sub[i], param))
        feats_arr = np.array(feats_list,dtype=np.float32)
        clf = LogisticRegression(max_iter=150, random_state=SEED)
        try:
            clf.fit(feats_arr, y_sub)
            preds = clf.predict(feats_arr)
            acc = accuracy_score(y_sub, preds)
        except:
            acc = 0.0
        fitnesses.append(acc)
    return fitnesses

def evolve_fractals(
    X_data,
    y_data,
    pop_size=15,
    generations=3,
    evolution_history=None,
    lod_name="LOD14",
    num_transforms=4
):
    """
    Evolve fractal params quickly:
    - pop_size=15
    - generations=3
    - partial classification on subset_size=500
    We'll store the best param each generation in history for 3D plotting.
    We'll also store top 3 or so at the end for ensemble usage.
    """
    # init pop
    population=[]
    for _ in range(pop_size):
        population.append(clamp_params(random_fractal_params(num_transforms)))

    fitnesses = evaluate_fractal_population(population, X_data, y_data, subset_size=500)
    best_idx = np.argmax(fitnesses)
    best_fit = fitnesses[best_idx]
    best_param = population[best_idx]

    if evolution_history is not None:
        evolution_history.append((f"{lod_name}_gen0", best_param))

    for g in range(generations):
        new_pop=[]
        while len(new_pop)<pop_size:
            cand_idx = np.random.choice(pop_size, 2, replace=False)
            f1=fitnesses[cand_idx[0]]
            f2=fitnesses[cand_idx[1]]
            if f1>f2:
                parent1=population[cand_idx[0]]
            else:
                parent1=population[cand_idx[1]]

            cand_idx2 = np.random.choice(pop_size, 2, replace=False)
            f3=fitnesses[cand_idx2[0]]
            f4=fitnesses[cand_idx2[1]]
            if f3>f4:
                parent2=population[cand_idx2[0]]
            else:
                parent2=population[cand_idx2[1]]

            c1,c2 = crossover(parent1, parent2)
            c1=mutate(c1,0.3)
            c2=mutate(c2,0.3)
            new_pop.append(c1)
            if len(new_pop)<pop_size:
                new_pop.append(c2)

        population=new_pop
        fitnesses = evaluate_fractal_population(population, X_data, y_data, subset_size=500)
        gen_best_idx = np.argmax(fitnesses)
        gen_best_fit = fitnesses[gen_best_idx]
        gen_best_param = population[gen_best_idx]
        if gen_best_fit>best_fit:
            best_fit=gen_best_fit
            best_param=gen_best_param
        print(f"   [Gen {g+1}] best fitness={best_fit:.4f}")
        if evolution_history is not None:
            evolution_history.append((f"{lod_name}_gen{g+1}", best_param))

    # We'll also pick top 3 for ensemble usage
    sorted_pop = sorted(zip(population,fitnesses), key=lambda x:x[1], reverse=True)
    topN = [p for (p,fit) in sorted_pop[:3]]
    return best_param, best_fit, topN

# ============================
# 3) PROGRESSIVE STAGES (LOD14->LOD28)
# ============================

def progressive_fractal_train(
    X_train,
    y_train,
    evolution_history=None
):
    """
    We'll do 2 LOD stages:
     - LOD14
     - LOD28
    Return an ensemble of top fractals from the final stage.
    """
    # Stage 1: LOD14
    print("\n--- LOD14 Stage ---")
    X_lod14 = downsample_images(X_train,(14,14))
    best_p14, best_fit14, top3_14 = evolve_fractals(
        X_lod14, y_train, pop_size=15,generations=3,
        evolution_history=evolution_history, lod_name="LOD14",num_transforms=4
    )
    print(f"LOD14 best_fit={best_fit14:.4f}")

    # Stage 2: LOD28
    print("\n--- LOD28 Stage ---")
    # We'll evolve again on full 28x28
    best_p28, best_fit28, top3_28 = evolve_fractals(
        X_train,y_train, pop_size=15,generations=3,
        evolution_history=evolution_history, lod_name="LOD28",num_transforms=4
    )
    print(f"LOD28 best_fit={best_fit28:.4f}")

    # We'll combine top fractals from both stages
    ensemble_fractals = top3_14 + top3_28
    return ensemble_fractals

def build_fractal_ensemble(ensemble_fractals, X_train, y_train):
    """
    We'll transform X_train into multiple sets of features (one per fractal param),
    and build an ensemble classifier using VotingClassifier.
    This can push accuracy significantly beyond a single fractal param.

    We'll do a quick logistic for each fractal -> Voting ensemble.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import VotingClassifier

    # build sub-estimators
    estimators=[]
    for i, param in enumerate(ensemble_fractals):
        # transform X_train
        feats=[]
        for img in X_train:
            feats.append(fractal_feature(img, param))
        feats_arr = np.array(feats,dtype=np.float32)
        # each sub-estimator is a pipeline of " identity transform -> logistic regression"
        # We'll just train it and store
        clf = LogisticRegression(max_iter=200, random_state=SEED+i)
        clf.fit(feats_arr, y_train)
        # We'll store param (f"fractal{i}")
        name_i = f"fr{i}"
        # but VotingClassifier expects an unfitted estimator unless we do hackish approach
        # Instead we do a small hack: we store the fitted logistic in a custom wrapper
        estimators.append((name_i, clf))

    # Now build a VotingClassifier with 'soft' voting if we had probabilities
    # But these are logistic => we can do 'soft'
    # We'll do 'hard' for simplicity
    voter = VotingClassifier(estimators=estimators, voting='hard')
    # We can't directly pass the fitted logistic into VotingClassifier elegantly
    # so let's do a hack: we won't re-fit them in the normal sense
    # We'll do a small trick: define a custom "predict" that uses the fitted one.
    # Simpler approach: let's define a wrapper class
    return estimators

def ensemble_predict(ensemble_estimators, X_test):
    """
    Manually do the "voting" from the sub-estimators we stored
    because we can't easily pass fitted classifiers into VotingClassifier.

    We'll gather predictions from each, do a majority vote.
    """
    from collections import Counter
    all_preds=[]
    for param_name, clf in ensemble_estimators:
        # transform X_test for that fractal
        feats_test = np.array([fractal_feature(img, clf.coef_.shape) for img in X_test])
        # Wait, we need param to transform. We don't have param, we just have the fitted logistic.
        # We have to store fractal param somewhere. We must store param in the name or define a bigger structure.

    # Instead, we define a bigger structure: let's store (param, logistic) in a custom list
    pass  # We'll fix this in final code below

# ============================
# 4) VISUALIZATION in 3D
# ============================

def generate_fractal_points_2d(fractal_params, n_points=300, seed=SEED):
    rng = np.random.default_rng(seed)
    x,y = 0.0,0.0
    transform_count = fractal_params.shape[0]
    points=[]
    for i in range(n_points):
        idx = rng.integers(transform_count)
        a,b,c,d,e,f = fractal_params[idx]
        x_new = a*x + b*y + c
        y_new = d*x + e*y + f
        x,y = x_new,y_new
        points.append((x,y))
    return np.array(points,dtype=np.float32)

def visualize_fractal_progress(evolution_history, points_per_param=300):
    """
    We'll do a 3D scatter where each generation's fractal is plotted at z=gen_index.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    for gen_index,(label,param) in enumerate(evolution_history):
        fractal_pts = generate_fractal_points_2d(param, n_points=points_per_param, seed=SEED+gen_index)
        x_vals = fractal_pts[:,0]
        y_vals = fractal_pts[:,1]
        z_vals = np.full_like(x_vals, gen_index,dtype=float)
        ax.scatter(x_vals, y_vals, z_vals, s=4, alpha=0.6, label=(label if gen_index<1 else None))

    ax.set_title("Fractal Evolution in 3D (Z=Generation)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Generation Step")
    plt.show()

# ============================
# 5) FULL CODE
# ============================

def main():
    print("Loading MNIST...")
    (X_train_full, y_train_full), (X_test, y_test)=load_mnist_data()

    # For speed, let's keep smaller subset if you want
    # e.g. 20000 for training
    # We'll do it fully by default, but you can slice for speed:
    # X_train_full = X_train_full[:20000]
    # y_train_full = y_train_full[:20000]

    print("Split train/val for partial approach...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=SEED, stratify=y_train_full
    )
    # We'll combine them after evolution
    X_trf = np.concatenate([X_train, X_val],axis=0)
    y_trf = np.concatenate([y_train, y_val],axis=0)

    # We'll store param evolution in a list
    evolution_history = []

    # 1) progressive fractal training
    start_t = time.time()
    print("=== Progressive Training with 2 LODs (14->28) ===")
    # LOD14
    print("\n--- LOD14 ---")
    X_lod14 = downsample_images(X_trf, (14,14))
    _,_, top3_14 = evolve_fractals(
        X_lod14, y_trf, pop_size=15,generations=3,
        evolution_history=evolution_history, lod_name="LOD14", num_transforms=4
    )

    # LOD28
    print("\n--- LOD28 ---")
    best_p28, best_fit28, top3_28 = evolve_fractals(
        X_trf,y_trf, pop_size=15,generations=3,
        evolution_history=evolution_history, lod_name="LOD28", num_transforms=4
    )

    # final ensemble fractals
    ensemble_fractals = top3_14 + top3_28
    print(f"LOD28 best fitness = {best_fit28:.4f}")
    train_time = time.time()-start_t
    print(f"Progressive fractal training took {train_time:.2f}s")

    # 2) Build final ensemble
    # We'll store (param, classifier). We'll do a straightforward approach:
    from sklearn.linear_model import LogisticRegression

    final_ensemble=[]
    print("\n[Final Ensemble] Fitting sub-classifiers for each fractal param in ensemble...")
    # We'll train on the entire X_trf again with each fractal param
    for i, param in enumerate(ensemble_fractals):
        feats=[]
        for img in X_trf:
            feats.append(fractal_feature(img, param))
        feats_arr = np.array(feats, dtype=np.float32)
        clf = LogisticRegression(max_iter=200, random_state=SEED+i)
        clf.fit(feats_arr, y_trf)
        final_ensemble.append((param, clf))

    # 3) Evaluate on test
    print("[Final] Evaluate on test set with majority vote ensemble")
    def ensemble_predict(X):
        # for each param, generate feats, do clf.predict
        # do majority vote
        all_preds=[]
        for (param,clf) in final_ensemble:
            feats_test = [fractal_feature(img, param) for img in X]
            feats_test_arr = np.array(feats_test,dtype=np.float32)
            pred_i = clf.predict(feats_test_arr)
            all_preds.append(pred_i)
        # shape => (#estimators, #samples)
        all_preds = np.array(all_preds)
        # majority vote
        final_preds=[]
        for col in all_preds.T:
            vals,counts = np.unique(col, return_counts=True)
            maj = vals[np.argmax(counts)]
            final_preds.append(maj)
        return np.array(final_preds,dtype=np.int32)

    t0=time.time()
    test_preds = ensemble_predict(X_test)
    test_time = time.time()-t0
    acc = accuracy_score(y_test,test_preds)
    print(f"[Final] Test accuracy = {acc:.4f}")
    print(f"Test inference time = {test_time:.4f}s")

    # 4) Visualization in 3D
    print("\nVisualizing fractal evolution in 3D...")
    visualize_fractal_progress(evolution_history, points_per_param=300)
    print(f"\nAll done. Final test accuracy = {acc:.4f}")

if __name__=="__main__":
    main()
