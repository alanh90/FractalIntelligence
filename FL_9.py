#!/usr/bin/env python3
"""
test_fractal_vs_nn_fashion_mnist_multiscale.py

Compares a multi-scale fractal dimension feature approach vs. a basic neural network (MLP)
on the Fashion-MNIST dataset.

Author: You
"""

import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras import models, layers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------------------------------------------------------------
# 1) MULTI-SCALE FRACTAL DIMENSION FEATURE EXTRACTOR FOR 28x28 IMAGES
# ----------------------------------------------------------------------------

class MultiScaleFractalFeatures:
    """
    A more advanced approach for 2D images:
      1) Convert each image to (28,28) float array => multiple thresholds => binarize
      2) For each binarized image, compute fractal dimension (box-counting)
      3) Also gather bounding box geometry or region count
      4) Combine across multiple thresholds => final feature vector
    This yields ~20–30 features per image.
    """

    def __init__(self, thresholds=[50, 100, 150, 200], box_sizes=[1,2,4,7,14], seed=SEED):
        """
        :param thresholds: list of intensity thresholds (0-255) to binarize
        :param box_sizes: scales used for box-counting fractal dimension
        """
        self.thresholds = thresholds
        self.box_sizes = box_sizes
        np.random.seed(seed)

    def fit(self, X):
        """
        Nothing to precompute globally.
        (But we define for consistency with scikit-like usage.)
        """
        pass

    def transform(self, X):
        """
        X is expected to be shape (n_samples, 784) for 28x28 images (flattened).
        We'll unflatten => (28,28) then process multiple thresholds => fractal features
        Return shape (n_samples, n_features).
        """
        n_samples = X.shape[0]
        feature_list = []

        for i in range(n_samples):
            img_flat = X[i]
            # reshape to 28x28
            img_2d = img_flat.reshape(28, 28)

            feat_vec = []
            for thr in self.thresholds:
                # binarize
                bin_img = (img_2d >= thr).astype(np.uint8)
                # compute fractal dimension
                fd = self._fractal_dimension_boxcount(bin_img, self.box_sizes)
                # bounding box geometry, region count, etc.
                geom = self._simple_geometry_features(bin_img)
                # add them all
                feat_vec.append(fd)
                feat_vec.extend(geom)
            # combine across all thresholds => single vector
            feature_list.append(feat_vec)

        return np.array(feature_list, dtype=np.float32)

    def _fractal_dimension_boxcount(self, bin_img, scales):
        """
        Basic box-counting fractal dimension for a binarized 2D image.
        :param bin_img: shape (28,28) binary
        :param scales: list of box sizes
        :return: fractal dimension (float)
        """
        # transform the bin_img so 1=occupied, 0=empty
        # We'll count how many boxes are needed to cover the 'occupied' pixels at each scale
        counts = []
        for s in scales:
            # how many boxes of size s x s needed
            n_boxes = 0
            step = s
            for row in range(0, 28, step):
                for col in range(0, 28, step):
                    # check if there's any '1' in this sub-box
                    sub = bin_img[row:row+step, col:col+step]
                    if np.any(sub == 1):
                        n_boxes += 1
            counts.append(max(n_boxes, 1))  # avoid zero

        # fractal dimension = slope in log-log space
        # x = log(1/scale), y = log(count)
        # scale factor for each s is (28 / s) in a sense, but let's just do a linear regression
        # We'll do log(counts) vs log(1.0/scales).
        scales_arr = np.array(scales, dtype=np.float32)
        counts_arr = np.array(counts, dtype=np.float32)
        log_counts = np.log(counts_arr)
        log_scales = np.log(1.0 / scales_arr)

        # linear regression slope
        n = len(scales_arr)
        meanX = np.mean(log_scales)
        meanY = np.mean(log_counts)
        num = 0.0
        den = 0.0
        for i in range(n):
            dx = log_scales[i] - meanX
            dy = log_counts[i] - meanY
            num += dx * dy
            den += dx * dx
        if abs(den) < 1e-12:
            return 0.0
        slope = num / den
        return slope  # fractal dimension approx

    def _simple_geometry_features(self, bin_img):
        """
        E.g., bounding box, # of '1' pixels, # of separate connected regions
        We'll do a quick approach with region counting via BFS or measure bounding box coverage
        """
        # bounding box
        coords = np.argwhere(bin_img==1)
        if len(coords)==0:
            return [0.0, 0.0, 0.0]  # no coverage => bounding box=0, region_count=0
        row_min = np.min(coords[:,0])
        row_max = np.max(coords[:,0])
        col_min = np.min(coords[:,1])
        col_max = np.max(coords[:,1])
        area = float(len(coords)) / (28*28)  # fraction of pixel coverage
        bbox_size = ((row_max-row_min+1)*(col_max-col_min+1)) / (28*28)
        region_count = self._count_regions(bin_img)

        return [area, bbox_size, region_count]

    def _count_regions(self, bin_img):
        """
        Count connected components of '1' in the bin_img using BFS/DFS
        """
        visited = np.zeros_like(bin_img, dtype=np.bool_)
        directions = [(1,0),(-1,0),(0,1),(0,-1)]
        rows, cols = bin_img.shape
        region_count = 0

        for r in range(rows):
            for c in range(cols):
                if bin_img[r,c]==1 and not visited[r,c]:
                    # start BFS
                    region_count += 1
                    stack = [(r,c)]
                    visited[r,c] = True
                    while stack:
                        rr, cc = stack.pop()
                        for dr,dc in directions:
                            nr, nc = rr+dr, cc+dc
                            if 0<=nr<rows and 0<=nc<cols:
                                if bin_img[nr,nc]==1 and not visited[nr,nc]:
                                    visited[nr,nc] = True
                                    stack.append((nr,nc))
        return float(region_count)


# ----------------------------------------------------------------------------
# 2) TRAIN FRACTAL FEATURES + KNN
# ----------------------------------------------------------------------------

def run_fractal_approach(X_train, y_train, X_test, y_test):
    # We'll do multi-scale fractal dimension for each image
    fractal_feat = MultiScaleFractalFeatures(
        thresholds=[50,100,150,200],
        box_sizes=[1,2,4,7,14],
        seed=SEED
    )
    print("\n[Fractal] Fitting fractal featurizer (nothing big to do, but consistent with pipeline)...")
    start = time.time()
    fractal_feat.fit(X_train)  # no-op in this approach
    fractal_fit_time = time.time() - start

    print("[Fractal] Transforming data => fractal features (this might take a bit)...")
    start = time.time()
    X_train_frac = fractal_feat.transform(X_train)
    X_test_frac  = fractal_feat.transform(X_test)
    fractal_transform_time = time.time() - start

    print("[Fractal] Feature shapes:", X_train_frac.shape, X_test_frac.shape)

    # scale these fractal features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_frac = scaler.fit_transform(X_train_frac)
    X_test_frac  = scaler.transform(X_test_frac)

    # KNN
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    print("[Fractal] Training KNN on fractal features...")
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_frac, y_train)
    knn_fit_time = time.time() - start

    start = time.time()
    y_pred = knn.predict(X_test_frac)
    knn_pred_time = time.time() - start

    fractal_bal_acc = balanced_accuracy_score(y_test, y_pred)
    fractal_acc = accuracy_score(y_test, y_pred)

    total_fractal_train = fractal_fit_time + fractal_transform_time + knn_fit_time
    fractal_predict_time = knn_pred_time

    print(f"[Fractal] Balanced Accuracy = {fractal_bal_acc:.4f}, Accuracy = {fractal_acc:.4f}")
    print(f"[Fractal] Total Train Time = {total_fractal_train:.4f}s, Predict Time = {fractal_predict_time:.4f}s")
    return fractal_bal_acc, fractal_acc, total_fractal_train, fractal_predict_time


# ----------------------------------------------------------------------------
# 3) BASIC MLP
# ----------------------------------------------------------------------------

def run_neural_network(X_train, y_train, X_test, y_test):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("\n[NN] Training MLP for 5 epochs...")
    start = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=5, batch_size=128, verbose=1, validation_split=0.1
    )
    nn_train_time = time.time() - start

    print("[NN] Predicting MLP on test data...")
    start = time.time()
    y_prob = model.predict(X_test)
    nn_predict_time = time.time() - start

    y_pred = np.argmax(y_prob, axis=1)
    nn_bal_acc = balanced_accuracy_score(y_test, y_pred)
    nn_acc = accuracy_score(y_test, y_pred)

    print(f"[NN] Balanced Accuracy = {nn_bal_acc:.4f}, Accuracy = {nn_acc:.4f}")
    print(f"[NN] Total Train Time = {nn_train_time:.4f}s, Predict Time = {nn_predict_time:.4f}s")

    return nn_bal_acc, nn_acc, nn_train_time, nn_predict_time


# ----------------------------------------------------------------------------
# 4) MAIN
# ----------------------------------------------------------------------------

def main():
    print("=== Loading Fashion-MNIST ===")
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

    # Flatten from (28,28) => (784)
    X_train_full = X_train_full.reshape(-1, 784).astype(np.float32)
    X_test       = X_test.reshape(-1, 784).astype(np.float32)

    # Quick train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=SEED, stratify=y_train_full
    )

    # Scale input to 0..1 for the MLP approach
    X_train_nn = X_train / 255.0
    X_val_nn   = X_val   / 255.0
    X_test_nn  = X_test  / 255.0

    # Combine train+val for final MLP training
    X_nn_final = np.concatenate([X_train_nn, X_val_nn], axis=0)
    y_nn_final = np.concatenate([y_train, y_val], axis=0)

    # 1) Fractal Approach: We'll just pass the integer-based arrays (0..255) or float(0..255)
    #    The fractal method does thresholding itself, so no strict normalization needed.
    X_trf = np.concatenate([X_train, X_val], axis=0)
    y_trf = np.concatenate([y_train, y_val], axis=0)

    print("\n=== Running Multi-Scale Fractal Approach ===")
    fractal_bal_acc, fractal_acc, fractal_train_time, fractal_pred_time = run_fractal_approach(
        X_trf, y_trf, X_test, y_test
    )

    print("\n=== Running Basic Neural Network (MLP) ===")
    nn_bal_acc, nn_acc, nn_train_time, nn_pred_time = run_neural_network(
        X_nn_final, y_nn_final, X_test_nn, y_test
    )

    print("\n===== FINAL RESULTS =====")
    print(f"Fractal Balanced Accuracy = {fractal_bal_acc:.4f}, Accuracy = {fractal_acc:.4f}")
    print(f"Fractal Training Time = {fractal_train_time:.4f}s, Prediction Time = {fractal_pred_time:.4f}s")
    print(f"NN Balanced Accuracy = {nn_bal_acc:.4f}, Accuracy = {nn_acc:.4f}")
    print(f"NN Training Time = {nn_train_time:.4f}s, Predict Time = {nn_pred_time:.4f}s")


if __name__ == "__main__":
    main()
