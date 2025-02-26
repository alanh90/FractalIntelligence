import numpy as np

try:
    import cupy as cp

    has_cupy = True
except ImportError:
    has_cupy = False
    cp = np  # Fallback to NumPy if CuPy is not available
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import time
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# --- Fast Fractal Flow Network --- #
class FastFractalFlowNetwork:
    def __init__(self, max_depth=3, threshold=0.1, use_gpu=True):
        """Initialize the Fast FFN."""
        self.max_depth = max_depth
        self.threshold = threshold
        self.use_gpu = use_gpu and has_cupy
        self.top_features = None
        self.medians = None
        self.majority_labels = None
        self.error_counts = None

    def fit(self, X, y):
        """Fit the model with binary splits on top features."""
        if self.use_gpu:
            X = cp.array(X)
            y = cp.array(y)
            feature_importance = mutual_info_classif(X.get(), y.get())
            feature_importance = cp.array(feature_importance)
            self.top_features = cp.argsort(-feature_importance)[:self.max_depth].get()  # Convert to NumPy for indexing
            self.medians = cp.median(X[:, self.top_features], axis=0)  # Keep as CuPy
        else:
            feature_importance = mutual_info_classif(X, y)
            self.top_features = np.argsort(-feature_importance)[:self.max_depth]
            self.medians = np.median(X[:, self.top_features], axis=0)

        # Compute bucket labels using binary splits on CPU
        X_cpu = X.get() if self.use_gpu else X
        binary_splits = (X_cpu[:, self.top_features] > self.medians.get() if self.use_gpu else self.medians).astype(int)
        bucket_labels = np.dot(binary_splits, (2 ** np.arange(self.max_depth))[::-1])

        # Compute majority labels for each bucket
        self.majority_labels = []
        for i in range(2 ** self.max_depth):
            mask = bucket_labels == i
            if np.sum(mask) > 0:
                values = y.get()[mask] if self.use_gpu else y[mask]
                majority = np.bincount(values).argmax()
            else:
                majority = 0
            self.majority_labels.append(majority)

        # Initialize error counts for adaptation
        self.error_counts = np.zeros(2 ** self.max_depth)

    def predict(self, X, y=None, adapt=False):
        """Predict labels for input data, with optional adaptation."""
        if self.use_gpu:
            X = cp.array(X)

        # Compute binary splits directly on X
        binary_splits = (X[:, self.top_features] > self.medians).astype(int)
        bucket_labels = cp.dot(binary_splits, (2 ** cp.arange(self.max_depth))[::-1]) if self.use_gpu else np.dot(binary_splits, (2 ** np.arange(self.max_depth))[::-1])

        # Convert bucket_labels to CPU for prediction and adaptation
        bucket_labels_cpu = bucket_labels.get() if self.use_gpu else bucket_labels
        preds = [self.majority_labels[int(label)] for label in bucket_labels_cpu]

        # Adapt the model if true labels are provided and adapt is True
        if adapt and y is not None:
            # Ensure y_cpu is a NumPy array
            y_cpu = y.get() if isinstance(y, cp.ndarray) else y
            for i, (pred, true) in enumerate(zip(preds, y_cpu)):
                if pred != true:
                    bucket = int(bucket_labels_cpu[i])
                    self.error_counts[bucket] += 1
                    if self.error_counts[bucket] / 10 > self.threshold:
                        self.majority_labels[bucket] = true
                        self.error_counts[bucket] = 0

        return np.array(preds)


# --- Dataset Loader --- #
def load_dataset(name):
    """Load a standard dataset and split it into training and testing sets."""
    if name == "Iris":
        data = load_iris()
    elif name == "Wine":
        data = load_wine()
    elif name == "Breast Cancer":
        data = load_breast_cancer()
    else:
        raise ValueError(f"Unknown dataset: {name}")

    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


# --- Model Evaluation --- #
def evaluate_models(dataset_name, X_train, X_test, y_train, y_test):
    """Evaluate the FastFractalFlowNetwork against Decision Tree and XGBoost."""
    print(f"\nEvaluating {dataset_name}")

    # Fast Fractal Flow Network
    ffn = FastFractalFlowNetwork(max_depth=3, threshold=0.1, use_gpu=has_cupy)
    start = time.time()
    ffn.fit(X_train, y_train)
    train_time = time.time() - start
    y_pred = ffn.predict(X_test, y_test, adapt=True)
    acc = accuracy_score(y_test, y_pred)
    print(f"Fast FFN - Time: {train_time:.4f}s, Accuracy: {acc:.4f}")

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    start = time.time()
    dt.fit(X_train, y_train)
    train_time = time.time() - start
    y_pred = dt.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Decision Tree - Time: {train_time:.4f}s, Accuracy: {acc:.4f}")

    # XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    start = time.time()
    xgb.fit(X_train, y_train)
    train_time = time.time() - start
    y_pred = xgb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"XGBoost - Time: {train_time:.4f}s, Accuracy: {acc:.4f}")


# --- Main Execution --- #
if __name__ == "__main__":
    datasets = ["Iris", "Wine", "Breast Cancer"]
    for dataset in datasets:
        X_train, X_test, y_train, y_test = load_dataset(dataset)
        evaluate_models(dataset, X_train, X_test, y_train, y_test)