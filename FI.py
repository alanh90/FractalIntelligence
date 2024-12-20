'''
================================================================================
SUMMARY OF FI RESULTS
================================================================================
      Dataset  FI Balanced Accuracy  FI Total Training Time (s)  FI Prediction Time (s)
        Moons              0.710000                      0.1310                  0.2235
      Circles              0.875000                      0.1312                  0.0100
       Digits              0.170568                      0.2171                  0.0135
         Iris              0.700000                      0.0400                  0.0090
Breast Cancer              0.871032                      0.0817                  0.0110
         Wine              0.619841                      0.0415                  0.0090
'''

import numpy as np
import time
from sklearn.datasets import make_moons, make_circles, load_digits, load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
from scipy.stats import moment


class FastFractalFeatures:
    def __init__(self, num_transformations=3, num_points=5000, seed=42):
        """
        A fast fractal-based feature extractor that:
        - Precomputes fractal patterns once at training.
        - Quickly adjusts patterns at inference.
        - Extracts simple geometric features.
        """
        self.num_transformations = num_transformations
        self.num_points = num_points
        self.seed = seed
        self.transformations = None
        self.base_fractal = None
        self.refined_fractal = None

    def _create_transformation(self, m1, m2, m3, m4):
        # Global transformations derived from dataset-level statistics
        a = 0.5 * np.tanh(m1 + 0.2 * m2)
        b = 0.3 * np.tanh(m3)
        c = 0.3 * np.tanh(m4)
        d = 0.5 * np.tanh(m1 - 0.2 * m2)
        e = 0.1 * m1
        f = 0.1 * m2
        return (a, b, e, c, d, f)

    def fit(self, X):
        np.random.seed(self.seed)
        # Compute global statistics
        m1 = np.mean(X)
        m2 = np.std(X)
        m3 = moment(X, moment=3, axis=None)
        m4 = moment(X, moment=4, axis=None)

        # Create a global set of transformations
        self.transformations = []
        for _ in range(self.num_transformations):
            self.transformations.append(self._create_transformation(m1, m2, m3, m4))
        self.transformations = np.array(self.transformations)

        # Pre-generate the base fractal pattern
        self.base_fractal = self._generate_fractal_points(self.transformations)

        # Optionally, create a slightly refined fractal pattern (multi-level)
        # For simplicity, we just perturb transformations slightly.
        refined_transformations = self.transformations.copy()
        refined_transformations += np.random.normal(0, 0.01, refined_transformations.shape)
        self.refined_fractal = self._generate_fractal_points(refined_transformations)

    def _generate_fractal_points(self, transformations):
        x, y = 0.0, 0.0
        points = []
        for _ in range(self.num_points):
            t_index = np.random.randint(len(transformations))
            t = transformations[t_index]
            x_new = t[0] * x + t[1] * y + t[2]
            y_new = t[3] * x + t[4] * y + t[5]
            x, y = np.clip(x_new, -1e10, 1e10), np.clip(y_new, -1e10, 1e10)
            points.append((x, y))
        return np.array(points)

    def transform(self, sample):
        """
        Transform a single sample into fractal-based features:
        - Use sample features (mean, std) to scale/shift the precomputed fractal patterns.
        - Extract simple geometric features from both base and refined fractals and combine.
        """
        # Simple scalar adjustments: we take mean and std of the sample features
        # and use them to scale and shift the precomputed fractal patterns.
        sample_mean = np.mean(sample)
        sample_std = np.std(sample) + 1e-9

        # Apply scaling/shift to base fractal
        transformed_base = self.base_fractal * sample_std + sample_mean
        transformed_refined = self.refined_fractal * (sample_std + 0.1) + (sample_mean - 0.05)

        # Extract geometric features from both sets and concatenate
        features_base = self._extract_geometric_features(transformed_base)
        features_refined = self._extract_geometric_features(transformed_refined)
        return np.concatenate([features_base, features_refined])

    @staticmethod
    def _extract_geometric_features(points):
        # Fast geometric features:
        # 1. Bounding box size
        # 2. Mean and variance of coordinates
        # 3. Min/Max distances from centroid

        if len(points) == 0:
            return np.zeros(10)

        x_coords = points[:, 0]
        y_coords = points[:, 1]

        xmin, xmax = np.min(x_coords), np.max(x_coords)
        ymin, ymax = np.min(y_coords), np.max(y_coords)
        width = xmax - xmin
        height = ymax - ymin

        x_mean = np.mean(x_coords)
        y_mean = np.mean(y_coords)
        x_var = np.var(x_coords)
        y_var = np.var(y_coords)

        # Distance from centroid
        dists = np.sqrt((x_coords - x_mean)**2 + (y_coords - y_mean)**2)
        d_min = np.min(dists)
        d_max = np.max(dists)
        d_mean = np.mean(dists)

        return np.array([width, height, x_mean, y_mean, x_var, y_var, d_min, d_max, d_mean, len(points)])


def test_on_dataset_fi(name: str, X: np.ndarray, y: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Training (fit) is just computing global stats and pre-generating fractals once
    start_time = time.time()
    fractal_model = FastFractalFeatures(num_transformations=3, num_points=2000)
    fractal_model.fit(X_train)
    training_time = time.time() - start_time

    print(f"Feature extraction (training setup) time: {training_time:.4f} seconds")

    # Transform each sample: This is now just a fast operation (scaling + a few arithmetic ops)
    start_time = time.time()
    X_train_fractal = np.array([fractal_model.transform(x) for x in X_train])
    X_test_fractal = np.array([fractal_model.transform(x) for x in X_test])
    feature_extraction_time = time.time() - start_time
    print(f"Feature extraction (per-sample) time: {feature_extraction_time:.4f} seconds")

    # KNN classification
    start_time = time.time()
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_fractal, y_train)
    training_classifier_time = time.time() - start_time
    print(f"Classifier training time: {training_classifier_time:.4f} seconds")

    start_time = time.time()
    predictions = knn.predict(X_test_fractal)
    prediction_time = time.time() - start_time
    print(f"Prediction time: {prediction_time:.4f} seconds")

    balanced_acc = balanced_accuracy_score(y_test, predictions)
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    return balanced_acc, training_time + feature_extraction_time + training_classifier_time, prediction_time


def run_experiments_fi():
    datasets = {
        "Moons": lambda: make_moons(n_samples=1000, noise=0.2, random_state=42),
        "Circles": lambda: make_circles(n_samples=1000, noise=0.2, factor=0.5, random_state=42),
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
