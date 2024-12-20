import numpy as np
import time
from sklearn.datasets import make_moons, make_circles, load_digits, load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
from scipy.stats import moment


class ClassLevelFractals:
    def __init__(self, num_transformations=3, num_points=2000, seed=42):
        """
        Class-level fractal feature extractor:
        - Compute fractal transformations per class.
        - Precompute fractal patterns (base + refined) per class.
        - At inference, find nearest class centroid and use that class's fractal pattern.
        """
        self.num_transformations = num_transformations
        self.num_points = num_points
        self.seed = seed
        self.class_transformations = {}
        self.class_base_fractals = {}
        self.class_refined_fractals = {}
        self.class_centroids = {}
        self.classes_ = None

    def _create_transformation(self, m1, m2, m3, m4):
        # Class-level transformations derived from class-level statistics
        a = 0.5 * np.tanh(m1 + 0.2 * m2)
        b = 0.3 * np.tanh(m3)
        c = 0.3 * np.tanh(m4)
        d = 0.5 * np.tanh(m1 - 0.2 * m2)
        e = 0.1 * m1
        f = 0.1 * m2
        return (a, b, e, c, d, f)

    def fit(self, X, y):
        np.random.seed(self.seed)
        self.classes_ = np.unique(y)

        # Compute class centroids
        for cls in self.classes_:
            class_samples = X[y == cls]
            centroid = np.mean(class_samples, axis=0)
            self.class_centroids[cls] = centroid

            # Compute class-level stats
            m1 = np.mean(class_samples)
            m2 = np.std(class_samples)
            m3 = moment(class_samples, moment=3, axis=None)
            m4 = moment(class_samples, moment=4, axis=None)

            # Create transformations for this class
            transformations = []
            for _ in range(self.num_transformations):
                transformations.append(self._create_transformation(m1, m2, m3, m4))
            transformations = np.array(transformations)
            self.class_transformations[cls] = transformations

            # Generate base fractal
            base_fractal = self._generate_fractal_points(transformations)
            self.class_base_fractals[cls] = base_fractal

            # Generate refined fractal (slight perturbation)
            refined_transformations = transformations.copy() + np.random.normal(0, 0.01, transformations.shape)
            refined_fractal = self._generate_fractal_points(refined_transformations)
            self.class_refined_fractals[cls] = refined_fractal

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
        For a given sample:
        - Find the nearest class centroid
        - Take that class's fractal patterns
        - Scale/shift them based on sample mean/std
        - Extract geometric features from both base and refined fractals and combine
        """
        nearest_class = self._find_nearest_class(sample)

        # Extract the corresponding class fractals
        base_fractal = self.class_base_fractals[nearest_class]
        refined_fractal = self.class_refined_fractals[nearest_class]

        # Scale/shift patterns based on sample mean/std
        sample_mean = np.mean(sample)
        sample_std = np.std(sample) + 1e-9

        transformed_base = base_fractal * sample_std + sample_mean
        transformed_refined = refined_fractal * (sample_std + 0.1) + (sample_mean - 0.05)

        features_base = self._extract_geometric_features(transformed_base)
        features_refined = self._extract_geometric_features(transformed_refined)
        return np.concatenate([features_base, features_refined])

    def _find_nearest_class(self, sample):
        # Find the class whose centroid is closest to this sample
        # This is a simple heuristic to pick the fractal pattern
        dists = {}
        for cls, centroid in self.class_centroids.items():
            d = np.linalg.norm(sample - centroid)
            dists[cls] = d
        return min(dists, key=dists.get)

    @staticmethod
    def _extract_geometric_features(points):
        # Simple geometric features from points
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

        dists = np.sqrt((x_coords - x_mean) ** 2 + (y_coords - y_mean) ** 2)
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

    # Fit fractal model
    start_time = time.time()
    fractal_model = ClassLevelFractals(num_transformations=3, num_points=2000)
    fractal_model.fit(X_train, y_train)
    training_setup_time = time.time() - start_time
    print(f"Feature extraction (training setup) time: {training_setup_time:.4f} seconds")

    # Transform train and test samples
    start_time = time.time()
    X_train_fractal = np.array([fractal_model.transform(x) for x in X_train])
    X_test_fractal = np.array([fractal_model.transform(x) for x in X_test])
    per_sample_feature_time = time.time() - start_time
    print(f"Feature extraction (per-sample) time: {per_sample_feature_time:.4f} seconds")

    start_time = time.time()
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_fractal, y_train)
    classifier_training_time = time.time() - start_time
    print(f"Classifier training time: {classifier_training_time:.4f} seconds")

    start_time = time.time()
    predictions = knn.predict(X_test_fractal)
    prediction_time = time.time() - start_time
    print(f"Prediction time: {prediction_time:.4f} seconds")

    balanced_acc = balanced_accuracy_score(y_test, predictions)
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    total_training_time = training_setup_time + per_sample_feature_time + classifier_training_time
    return balanced_acc, total_training_time, prediction_time


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
