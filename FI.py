import numpy as np
import time
from sklearn.datasets import make_moons, make_circles, load_digits, load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
import pandas as pd

class IFSFractal:
    def __init__(self, transformations):
        self.transformations = transformations

    def generate_points(self, num_points=1000):  # Reduced number of points
        x, y = 0.0, 0.0
        points = []
        for _ in range(num_points):
            t_index = np.random.randint(len(self.transformations))
            t = self.transformations[t_index]
            x_new = t[0] * x + t[1] * y + t[2]
            y_new = t[3] * x + t[4] * y + t[5]
            x, y = x_new, y_new

            # Clamping to prevent overflow
            x = np.clip(x, -1e10, 1e10) # Added clipping
            y = np.clip(y, -1e10, 1e10) # Added clipping

            points.append((x, y))
        return np.array(points)

    def extract_features(self, points):
        if len(points) == 0:
            return np.array([0, 0, 0, 0])
        x = points[:, 0]
        y = points[:, 1]
        return np.array([np.mean(x), np.std(x), np.mean(y), np.std(y)])


def create_fractal_features(X, num_transformations=3, num_points=1000):
    n_samples, n_features = X.shape
    fractal_features = []
    for sample in X:
        transformations = []
        for i in range(num_transformations):
            # Improved mapping with scaling and offsetting
            a = 0.5 * np.tanh(np.mean(sample) + np.std(sample) * 0.1 * (i+1)) # Using tanh and scaling
            b = 0.5 * np.tanh(np.mean(sample) + np.std(sample) * 0.05 * (i+2)) # Using tanh and scaling
            c = 0.5 * np.tanh(np.mean(sample) + np.std(sample) * 0.05 * (i+3)) # Using tanh and scaling
            d = 0.5 * np.tanh(np.mean(sample) + np.std(sample) * 0.1 * (i+4)) # Using tanh and scaling
            e = 0.1 * np.mean(sample) # Scaling
            f = 0.1 * np.std(sample) # Scaling

            transformations.append((a, b, e, c, d, f))
        fractal = IFSFractal(transformations)
        points = fractal.generate_points(num_points)
        features = fractal.extract_features(points)
        fractal_features.append(features)
    return np.array(fractal_features)


def test_on_dataset(name: str, X: np.ndarray, y: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    start_time = time.time()
    X_train_fractal = create_fractal_features(X_train)
    X_test_fractal = create_fractal_features(X_test)
    training_time = time.time() - start_time
    print(f"Fractal feature extraction time: {training_time:.4f} seconds")

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_fractal, y_train)

    start_time = time.time()
    predictions = knn.predict(X_test_fractal)
    prediction_time = time.time() - start_time
    print(f"Prediction time: {prediction_time:.4f} seconds")
    balanced_acc = balanced_accuracy_score(y_test, predictions)
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    return balanced_acc, training_time, prediction_time

def run_experiments():
    datasets = {
        "Moons": lambda: make_moons(n_samples=1000, noise=0.2, random_state=42),
        "Circles": lambda: make_circles(n_samples=1000, noise=0.2, factor=0.5, random_state=42),
        "Digits": lambda: load_digits(return_X_y=True),
        "Iris": lambda: load_iris(return_X_y=True),
        "Breast Cancer": lambda: load_breast_cancer(return_X_y=True),
        "Wine": lambda: load_wine(return_X_y=True),
    }
    results = []
    for dataset_name, dataset_loader in datasets.items():
        print(f"\nTesting on {dataset_name} dataset...")
        X, y = dataset_loader()
        balanced_acc, training_time, prediction_time = test_on_dataset(dataset_name, X, y)
        results.append(
                {
                    "Dataset": dataset_name,
                    "Balanced Accuracy": balanced_acc,
                    "Training Time (s)": round(training_time, 4),
                    "Prediction Time (s)": round(prediction_time, 4),
                }
            )
    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    run_experiments()