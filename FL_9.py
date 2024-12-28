#!/usr/bin/env python3
"""
test_fractal_vs_nn_fashion_mnist.py

Compares a simple fractal-based feature approach vs. a basic neural network (MLP)
on the Fashion-MNIST dataset.

Author: You
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras import models, layers
import time


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ----------------------------------------------------------------------------
# 1) SIMPLE FRACTAL FEATURE EXTRACTOR (fast demonstration)
# ----------------------------------------------------------------------------

class SimpleFractalFeaturizer:
    """
    A minimal fractal-based feature extractor demonstration:
      - We define a small set of transformations from dataset-level stats
      - Pre-generate fractal points
      - For each sample, we scale/shift those points based on sample stats
      - Extract bounding-box geometry => 10 features
      - Returns that feature vector
    """

    def __init__(self, num_points=1000, seed=SEED):
        self.num_points = num_points
        self.seed = seed
        self.transformations = None  # shape (N, 6)
        self.base_fractal = None

    def fit(self, X):
        # X is an array of shape (n_samples, 784) if flattened Fashion-MNIST
        np.random.seed(self.seed)
        # global stats
        global_mean = np.mean(X)
        global_std = np.std(X) + 1e-9

        # Create a small set of transformations based on these stats
        # For example, 2 transformations => shape (2, 6)
        self.transformations = []
        for i in range(2):
            a = 0.5 * np.tanh(global_mean + 0.2 * global_std)
            b = 0.3 * np.tanh(global_std)
            c = 0.1 * global_mean
            d = 0.5 * np.tanh(global_mean - 0.1 * global_std)
            e = 0.1 * global_std
            f = 0.05 * global_mean
            self.transformations.append([a, b, c, d, e, f])
        self.transformations = np.array(self.transformations)

        # Pre-generate the fractal points once
        self.base_fractal = self._generate_fractal_points(self.transformations)

    def _generate_fractal_points(self, transformations):
        # Very simplistic IFS generation
        x, y = 0.0, 0.0
        points = []
        np.random.seed(self.seed)
        for _ in range(self.num_points):
            idx = np.random.randint(len(transformations))
            a, b, c, d, e, f = transformations[idx]
            x_new = a * x + b * y + c
            y_new = d * x + e * y + f
            x, y = x_new, y_new
            points.append((x, y))
        return np.array(points)

    def _extract_geometry(self, points):
        # returns 10 features: bounding box + centroid + var + dist stats
        if len(points) == 0:
            return np.zeros(10)
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

        dists = np.sqrt((x_coords - xm) ** 2 + (y_coords - ym) ** 2)
        dmin = np.min(dists)
        dmax = np.max(dists)
        return np.array([width, height, xm, ym, xv, yv, dmin, dmax, np.mean(dists), len(points)], dtype=np.float32)

    def transform(self, X):
        """
        For each sample:
          1) take base_fractal
          2) scale by sample_mean/std
          3) extract geometry => returns shape (n_samples, 10)
        """
        feats = []
        for sample in X:
            sm = np.mean(sample)
            ss = np.std(sample) + 1e-9
            scaled_pts = self.base_fractal * ss + sm
            g = self._extract_geometry(scaled_pts)
            feats.append(g)
        return np.array(feats)


# ----------------------------------------------------------------------------
# 2) TRAIN A SIMPLE FRACTAL + KNN APPROACH
# ----------------------------------------------------------------------------

def run_fractal_approach(X_train, y_train, X_test, y_test):
    featurizer = SimpleFractalFeaturizer(num_points=1000, seed=SEED)

    print("\n[Fractal] Fitting fractal featurizer...")
    start = time.time()
    featurizer.fit(X_train)
    fractal_fit_time = time.time() - start

    print("[Fractal] Transforming data into fractal features...")
    start = time.time()
    X_train_frac = featurizer.transform(X_train)
    X_test_frac = featurizer.transform(X_test)
    fractal_transform_time = time.time() - start

    # train a small KNN
    print("[Fractal] Training KNN on fractal features...")
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_frac, y_train)
    knn_fit_time = time.time() - start

    # evaluate
    start = time.time()
    y_pred = knn.predict(X_test_frac)
    knn_pred_time = time.time() - start

    # balanced acc or normal accuracy
    fractal_bal_acc = balanced_accuracy_score(y_test, y_pred)
    fractal_acc = accuracy_score(y_test, y_pred)

    total_fractal_train = fractal_fit_time + fractal_transform_time + knn_fit_time
    fractal_predict_time = knn_pred_time

    print(f"[Fractal] Balanced Accuracy = {fractal_bal_acc:.4f}, Accuracy = {fractal_acc:.4f}")
    print(f"[Fractal] Total Train Time = {total_fractal_train:.4f}s, Predict Time = {fractal_predict_time:.4f}s")
    return fractal_bal_acc, fractal_acc, total_fractal_train, fractal_predict_time


# ----------------------------------------------------------------------------
# 3) TRAIN A BASIC NEURAL NETWORK (MLP)
# ----------------------------------------------------------------------------

def run_neural_network(X_train, y_train, X_test, y_test):
    """
    A minimal MLP for classification on flattened 28x28 images.
    """
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')  # 10 classes for Fashion-MNIST
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("\n[NN] Training MLP...")
    start = time.time()
    # We'll do a short training (5 epochs) for demonstration
    history = model.fit(X_train, y_train, epochs=5, batch_size=128, verbose=1, validation_split=0.1)
    nn_train_time = time.time() - start

    print("[NN] Predicting MLP on test data...")
    start = time.time()
    y_prob = model.predict(X_test)  # shape (n_test, 10)
    nn_predict_time = time.time() - start

    y_pred = np.argmax(y_prob, axis=1)
    nn_bal_acc = balanced_accuracy_score(y_test, y_pred)
    nn_acc = accuracy_score(y_test, y_pred)

    total_nn_train_time = nn_train_time
    print(f"[NN] Balanced Accuracy = {nn_bal_acc:.4f}, Accuracy = {nn_acc:.4f}")
    print(f"[NN] Total Train Time = {total_nn_train_time:.4f}s, Predict Time = {nn_predict_time:.4f}s")
    return nn_bal_acc, nn_acc, total_nn_train_time, nn_predict_time


# ----------------------------------------------------------------------------
# 4) MAIN: LOAD FASHION-MNIST, RUN FRACTAL & NN
# ----------------------------------------------------------------------------

def main():
    # load Fashion-MNIST from TensorFlow
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

    # Typically we take a smaller subset for demonstration or keep all data
    # We'll keep the entire dataset
    # Flatten the 28x28 images into 784
    X_train_full = X_train_full.reshape(-1, 784).astype(np.float32)
    X_test = X_test.reshape(-1, 784).astype(np.float32)

    # We can do a new train/val split or just trust the built-in test
    # Let's do a quick shuffle to create a smaller validation set:
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=SEED, stratify=y_train_full
    )

    # For fractal & MLP, let's scale to [0,1] or standard scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Combine train + val for final fractal approach or keep them separate
    X_trf = np.concatenate([X_train_s, X_val_s], axis=0)
    y_trf = np.concatenate([y_train, y_val], axis=0)

    # 1) Run fractal approach on (train+val) => test
    print("=== Running Fractal Approach ===")
    fractal_bal_acc, fractal_acc, fractal_train_time, fractal_pred_time = run_fractal_approach(
        X_trf, y_trf, X_test_s, y_test
    )

    # 2) Run MLP approach on (train+val) => test
    # We'll keep it simple: re-init model and train on (train+val)
    print("\n=== Running Basic Neural Network (MLP) ===")
    # For demonstration, pass (train+val) to the MLP
    nn_bal_acc, nn_acc, nn_train_time, nn_pred_time = run_neural_network(
        X_trf, y_trf, X_test_s, y_test
    )

    print("\n===== FINAL RESULTS =====")
    print(f"Fractal Balanced Accuracy = {fractal_bal_acc:.4f}, Accuracy = {fractal_acc:.4f}")
    print(f"Fractal Training Time = {fractal_train_time:.4f}s, Predict Time = {fractal_pred_time:.4f}s")

    print(f"NN Balanced Accuracy = {nn_bal_acc:.4f}, Accuracy = {nn_acc:.4f}")
    print(f"NN Training Time = {nn_train_time:.4f}s, Predict Time = {nn_pred_time:.4f}s")


if __name__ == "__main__":
    main()
