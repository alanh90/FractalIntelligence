"""

Output of this file:

C:\Users\ahour\PycharmProjects\FractalIntelligence\.venv\Scripts\python.exe C:\Users\ahour\PycharmProjects\FractalIntelligence\NN_v_FL_9.py
2024-12-28 18:33:28.433440: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2024-12-28 18:33:28.894832: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
Loading MNIST...
Splitting train/val for fractal approach... (Neural net will do an internal val split).
Train set shape: (60000, 28, 28), test shape: (10000, 28, 28)

--- LOD14 ---
   [LOD14 Gen 1] best_fit=0.2220
   [LOD14 Gen 2] best_fit=0.2220
   [LOD14 Gen 3] best_fit=0.2220

--- LOD28 ---
   [LOD28 Gen 1] best_fit=0.2480
   [LOD28 Gen 2] best_fit=0.2480
   [LOD28 Gen 3] best_fit=0.2480

[Fractal Approach] Progressive training time = 50.56s

[Fractal Approach] Building final ensemble on the entire X_trf...
Ensemble build time = 10.43s

[Fractal Approach] Evaluate final ensemble on test set...
[Fractal Approach] Test accuracy = 0.2257, test inference time=1.22s

Visualizing fractal progression in 3D...

=== Training Basic NN for Comparison ===
C:\Users\ahour\PycharmProjects\FractalIntelligence\.venv\lib\site-packages\keras\src\layers\core\dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
2024-12-28 18:34:42.481220: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

[NN] Training MLP for 5 epochs...
Epoch 1/5
422/422 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.8085 - loss: 0.6593 - val_accuracy: 0.9605 - val_loss: 0.1393
Epoch 2/5
422/422 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9537 - loss: 0.1586 - val_accuracy: 0.9705 - val_loss: 0.1035
Epoch 3/5
422/422 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9686 - loss: 0.1054 - val_accuracy: 0.9737 - val_loss: 0.0915
Epoch 4/5
422/422 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9772 - loss: 0.0776 - val_accuracy: 0.9745 - val_loss: 0.0872
Epoch 5/5
422/422 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9827 - loss: 0.0598 - val_accuracy: 0.9760 - val_loss: 0.0840
[NN] Predicting on test data...
313/313 ━━━━━━━━━━━━━━━━━━━━ 0s 596us/step
[NN] Final Test Acc=0.9721, TrainTime=3.51s, PredictTime=0.28s

===== FINAL COMPARISON =====
Fractal Approach: Acc=0.2257, TrainTime~61.00s, PredictTime~1.22s
Neural Net: Acc=0.9721, TrainTime=3.51s, PredictTime=0.28s

All done.

Process finished with exit code 0

"""

import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D scatter plot
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================
# 1) LOADING MNIST
# ============================

def load_mnist_data():
    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Convert to floats 0..1
    X_train_full = X_train_full.astype(np.float32) / 255.0
    X_test       = X_test.astype(np.float32)       / 255.0
    return (X_train_full, y_train_full), (X_test, y_test)

# ============================
# 2) DOWNSAMPLING
# ============================

def downsample_images(X, new_size=(14,14)):
    """
    Example: from 28x28 -> 14x14 via average pooling.
    """
    old_size = X.shape[1]
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
# 3) FRACTAL PARAMS & EVOLUTION
# ============================

def random_fractal_params(num_transforms=4):
    # shape (num_transforms, 6)
    return np.random.uniform(-1, 1, size=(num_transforms, 6))

def clamp_params(params):
    np.clip(params, -1.0, 1.0, out=params)
    return params

def mutate(params, rate=0.3):
    p_new = np.copy(params)
    if np.random.rand() < rate:
        r_i = np.random.randint(p_new.shape[0])
        c_i = np.random.randint(p_new.shape[1])
        p_new[r_i, c_i] += np.random.normal(0, 0.2)
    return clamp_params(p_new)

def crossover(p1, p2):
    row_pt = np.random.randint(p1.shape[0])
    c1 = np.vstack([p1[:row_pt], p2[row_pt:]])
    c2 = np.vstack([p2[:row_pt], p1[row_pt:]])
    clamp_params(c1)
    clamp_params(c2)
    return c1, c2

def fractal_feature(img, fractal_params):
    # Use (mean, std) -> expansions
    mean_val = np.mean(img)
    std_val  = np.std(img) + 1e-9

    feats=[]
    for row in fractal_params:
        a,b,c,d,e,f = row
        x_new = a*mean_val + b*std_val + c
        y_new = d*mean_val + e*std_val + f
        feats.extend([x_new, y_new])
    return np.array(feats, dtype=np.float32)

from sklearn.linear_model import LogisticRegression
def evaluate_fractal_population(
    population,
    X_data,
    y_data,
    subset_size=500,
):
    # Evaluate classification on random subset => logistic regression
    if len(X_data) < subset_size:
        subset_size = len(X_data)
    idx_sub = np.random.choice(len(X_data), subset_size, replace=False)
    X_sub = X_data[idx_sub]
    y_sub = y_data[idx_sub]

    fitnesses=[]
    for param in population:
        feats_list=[]
        for i in range(subset_size):
            feats_list.append(fractal_feature(X_sub[i], param))
        feats_arr = np.array(feats_list,dtype=np.float32)

        clf=LogisticRegression(max_iter=150, random_state=SEED)
        try:
            clf.fit(feats_arr, y_sub)
            preds=clf.predict(feats_arr)
            acc=accuracy_score(y_sub, preds)
        except:
            acc=0.0
        fitnesses.append(acc)
    return fitnesses

def evolve_fractals(
    X_data, y_data,
    pop_size=15,
    generations=3,
    lod_name="LOD14",
    num_transforms=4,
    subset_size=500,
    evolution_history=None
):
    # init
    population=[]
    for _ in range(pop_size):
        population.append(clamp_params(random_fractal_params(num_transforms)))
    fitnesses = evaluate_fractal_population(population, X_data, y_data, subset_size)
    best_idx = np.argmax(fitnesses)
    best_fit=fitnesses[best_idx]
    best_param=population[best_idx]

    # record gen0
    if evolution_history is not None:
        evolution_history.append((f"{lod_name}_gen0", best_param))

    for g in range(generations):
        new_pop=[]
        while len(new_pop)<pop_size:
            cand_idx = np.random.choice(pop_size,2,replace=False)
            f1=fitnesses[cand_idx[0]]
            f2=fitnesses[cand_idx[1]]
            if f1>f2:
                parent1=population[cand_idx[0]]
            else:
                parent1=population[cand_idx[1]]

            cand_idx2 = np.random.choice(pop_size,2,replace=False)
            f3=fitnesses[cand_idx2[0]]
            f4=fitnesses[cand_idx2[1]]
            if f3>f4:
                parent2=population[cand_idx2[0]]
            else:
                parent2=population[cand_idx2[1]]

            c1,c2=crossover(parent1, parent2)
            c1=mutate(c1,0.3)
            c2=mutate(c2,0.3)
            new_pop.append(c1)
            if len(new_pop)<pop_size:
                new_pop.append(c2)

        population=new_pop
        fitnesses = evaluate_fractal_population(population, X_data,y_data, subset_size)
        gen_best_idx=np.argmax(fitnesses)
        gen_best_fit=fitnesses[gen_best_idx]
        gen_best_param=population[gen_best_idx]
        if gen_best_fit>best_fit:
            best_fit=gen_best_fit
            best_param=gen_best_param
        print(f"   [{lod_name} Gen {g+1}] best_fit={best_fit:.4f}")

        if evolution_history is not None:
            evolution_history.append((f"{lod_name}_gen{g+1}", best_param))

    # pick top 3 for ensemble
    pop_fit = sorted(zip(population,fitnesses), key=lambda x:x[1], reverse=True)
    top3=[p for (p,f) in pop_fit[:3]]
    return best_param, best_fit, top3

# ============================
# 4) PROGRESSIVE
# ============================

def progressive_fractal_train(
    X_train,
    y_train,
    evolution_history=None
):
    print("\n--- LOD14 ---")
    X_lod14 = downsample_images(X_train,(14,14))
    best14,fit14,top3_14 = evolve_fractals(X_lod14, y_train, pop_size=15,generations=3,
                                          lod_name="LOD14",num_transforms=4,
                                          subset_size=500,
                                          evolution_history=evolution_history)

    print("\n--- LOD28 ---")
    best28,fit28,top3_28 = evolve_fractals(X_train, y_train, pop_size=15,generations=3,
                                          lod_name="LOD28",num_transforms=4,
                                          subset_size=500,
                                          evolution_history=evolution_history)
    ensemble_fractals = top3_14 + top3_28
    return ensemble_fractals

def build_fractal_ensemble(ensemble_fractals, X_train, y_train):
    """
    We'll transform X_train for each fractal param => train logistic => store (param, classifier).
    """
    from sklearn.linear_model import LogisticRegression
    final_ensemble=[]
    for i,param in enumerate(ensemble_fractals):
        feats_list=[]
        for img in X_train:
            feats_list.append(fractal_feature(img, param))
        feats_arr=np.array(feats_list,dtype=np.float32)

        clf=LogisticRegression(max_iter=200, random_state=SEED+i)
        clf.fit(feats_arr,y_train)
        final_ensemble.append((param,clf))
    return final_ensemble

def ensemble_predict(ensemble, X):
    """
    majority vote from each sub-classifier
    """
    from collections import Counter
    all_preds=[]
    for (param,clf) in ensemble:
        feats_list = [fractal_feature(img, param) for img in X]
        feats_arr = np.array(feats_list,dtype=np.float32)
        pred_i = clf.predict(feats_arr)
        all_preds.append(pred_i)

    all_preds = np.array(all_preds) # shape (#estimators, #samples)
    final_preds=[]
    for col in all_preds.T:
        vals,counts = np.unique(col, return_counts=True)
        maj = vals[np.argmax(counts)]
        final_preds.append(maj)
    return np.array(final_preds,dtype=int)

# ============================
# 5) VISUALIZE
# ============================

def generate_fractal_points_2d(params, n_points=300, seed=SEED):
    rng = np.random.default_rng(seed)
    x,y=0.0,0.0
    t_count=params.shape[0]
    pts=[]
    for i in range(n_points):
        idx = rng.integers(t_count)
        a,b,c,d,e,f = params[idx]
        x_new=a*x+b*y+c
        y_new=d*x+e*y+f
        x,y=x_new,y_new
        pts.append((x,y))
    return np.array(pts,dtype=np.float32)

def visualize_fractal_progress(evolution_history, points_per_param=300):
    fig=plt.figure(figsize=(8,6))
    ax=fig.add_subplot(111,projection='3d')

    for gen_index,(label,param) in enumerate(evolution_history):
        fractal_pts=generate_fractal_points_2d(param, n_points=points_per_param, seed=SEED+gen_index)
        x_vals=fractal_pts[:,0]
        y_vals=fractal_pts[:,1]
        z_vals=np.full_like(x_vals, gen_index,dtype=float)
        ax.scatter(x_vals,y_vals,z_vals,s=4,alpha=0.6,label=(label if gen_index<1 else None))

    ax.set_title("Fractal Evolution in 3D (Z=Generation)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Gen Step")
    plt.show()

# ============================
# 6) BASIC NEURAL NETWORK
# ============================

def train_neural_network(X_train, y_train, X_test, y_test):
    """
    A short Keras MLP: 2 dense layers + final 10-class softmax
    We'll train for 5 epochs to keep it short.
    We'll measure final test accuracy.
    """
    from tensorflow.keras import models, layers

    # Flatten the (28,28) -> (784)
    # If data is not 28x28, do it carefully or do a different shape
    # We'll assume it's full 28x28 for now
    X_train_flat = X_train.reshape(len(X_train), 28*28)
    X_test_flat  = X_test.reshape(len(X_test),   28*28)

    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("\n[NN] Training MLP for 5 epochs...")
    start=time.time()
    history = model.fit(
        X_train_flat, y_train,
        epochs=5, batch_size=128, verbose=1,
        validation_split=0.1
    )
    nn_train_time=time.time()-start

    print("[NN] Predicting on test data...")
    start=time.time()
    y_prob=model.predict(X_test_flat)
    nn_predict_time=time.time()-start

    y_pred=np.argmax(y_prob,axis=1)
    nn_acc=accuracy_score(y_test,y_pred)

    print(f"[NN] Final Test Acc={nn_acc:.4f}, TrainTime={nn_train_time:.2f}s, PredictTime={nn_predict_time:.2f}s")
    return nn_acc, nn_train_time, nn_predict_time


# ============================
# MAIN
# ============================

def main():
    print("Loading MNIST...")
    (X_train_full, y_train_full), (X_test, y_test) = load_mnist_data()

    print("Splitting train/val for fractal approach... (Neural net will do an internal val split).")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=SEED, stratify=y_train_full
    )
    # Combine after
    X_trf = np.concatenate([X_train, X_val], axis=0)
    y_trf = np.concatenate([y_train, y_val], axis=0)

    # Make sure X_test is also (28,28)
    # We'll do the fractal approach on 28x28 or downsample to 14
    print(f"Train set shape: {X_trf.shape}, test shape: {X_test.shape}")

    # 1) Progressive fractal approach
    evolution_history=[]
    start_t=time.time()
    ensemble_fractals = progressive_fractal_train(X_trf,y_trf, evolution_history=evolution_history)
    fractal_train_time = time.time()-start_t
    print(f"\n[Fractal Approach] Progressive training time = {fractal_train_time:.2f}s")

    # Build ensemble
    print("\n[Fractal Approach] Building final ensemble on the entire X_trf...")
    t0=time.time()
    final_ensemble = build_fractal_ensemble(ensemble_fractals, X_trf, y_trf)
    build_ens_time = time.time()-t0
    print(f"Ensemble build time = {build_ens_time:.2f}s")

    # Evaluate on test
    print("\n[Fractal Approach] Evaluate final ensemble on test set...")
    t0=time.time()
    fractal_preds = ensemble_predict(final_ensemble, X_test)
    fractal_test_time = time.time()-t0
    fractal_acc = accuracy_score(y_test, fractal_preds)
    print(f"[Fractal Approach] Test accuracy = {fractal_acc:.4f}, test inference time={fractal_test_time:.2f}s")

    # Visualize fractal evolution in 3D
    print("\nVisualizing fractal progression in 3D...")
    visualize_fractal_progress(evolution_history, points_per_param=300)

    # 2) Basic neural network for comparison
    print("\n=== Training Basic NN for Comparison ===")
    nn_start=time.time()
    nn_acc, nn_train_time, nn_pred_time = train_neural_network(X_train_full, y_train_full, X_test, y_test)
    total_nn_time=time.time()-nn_start

    # Summaries
    print("\n===== FINAL COMPARISON =====")
    print(f"Fractal Approach: Acc={fractal_acc:.4f}, TrainTime~{fractal_train_time+build_ens_time:.2f}s, PredictTime~{fractal_test_time:.2f}s")
    print(f"Neural Net: Acc={nn_acc:.4f}, TrainTime={nn_train_time:.2f}s, PredictTime={nn_pred_time:.2f}s")

    print("\nAll done.")

if __name__=="__main__":
    main()
