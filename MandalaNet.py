import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import time
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from scipy.stats import entropy as scipy_entropy


# --- Fast Fractal Flow Network with Entropy and Probabilistic Predictions --- #
class FastFractalFlowNetwork:
    def __init__(self, max_depth=3, threshold=0.1):
        """
        Initialize the FastFractalFlowNetwork.

        Parameters:
        - max_depth: Maximum fractal depth (number of features to split on).
        - threshold: Error threshold for triggering adaptation.
        """
        self.max_depth = max_depth
        self.threshold = threshold
        self.top_features = None  # Indices of selected features
        self.medians = None  # Median values for splitting
        self.bucket_probs = None  # Probability distributions per bucket
        self.error_counts = None  # Error tracking for adaptation

    def fit(self, X, y):
        """
        Fit the model to the training data.

        Parameters:
        - X: Feature matrix (n_samples, n_features).
        - y: Target array (n_samples).
        """
        # Select features using entropy
        entropies = [scipy_entropy(X[:, i]) for i in range(X.shape[1])]
        self.top_features = np.argsort(entropies)[-self.max_depth:]

        # Compute medians for splitting
        self.medians = np.median(X[:, self.top_features], axis=0)

        # Assign data to buckets using binary splits
        binary_splits = (X[:, self.top_features] > self.medians).astype(int)
        bucket_labels = np.dot(binary_splits, (2 ** np.arange(self.max_depth))[::-1])

        # Compute probability distributions for each bucket
        self.bucket_probs = []
        for i in range(2 ** self.max_depth):
            mask = bucket_labels == i
            if np.sum(mask) > 0:
                values = y[mask]
                counts = np.bincount(values, minlength=np.max(y) + 1)
                probs = counts / np.sum(counts)
            else:
                probs = np.zeros(np.max(y) + 1)
            self.bucket_probs.append(probs)

        # Initialize error counts
        self.error_counts = np.zeros(2 ** self.max_depth)

    def predict(self, X, y=None, adapt=False):
        """
        Predict labels for input data, with optional adaptation.

        Parameters:
        - X: Feature matrix (n_samples, n_features).
        - y: True labels (optional, for adaptation).
        - adapt: Boolean to enable online adaptation.

        Returns:
        - Predicted labels (n_samples).
        """
        # Compute bucket assignments
        binary_splits = (X[:, self.top_features] > self.medians).astype(int)
        bucket_labels = np.dot(binary_splits, (2 ** np.arange(self.max_depth))[::-1])

        # Predict using maximum probability
        preds = [np.argmax(self.bucket_probs[label]) for label in bucket_labels]

        # Adapt model if true labels provided
        if adapt and y is not None:
            for i, (pred, true) in enumerate(zip(preds, y)):
                if pred != true:
                    bucket = bucket_labels[i]
                    self.error_counts[bucket] += 1
                    if self.error_counts[bucket] / 10 > self.threshold:
                        # Update probabilities incrementally
                        self.bucket_probs[bucket][true] += 0.1
                        self.bucket_probs[bucket] /= np.sum(self.bucket_probs[bucket])
                        self.error_counts[bucket] = 0

        return np.array(preds)


# --- Dataset Loader --- #
def load_dataset(name):
    """Load and preprocess a dataset."""
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
    """Evaluate FFN, Decision Tree, and XGBoost on a dataset."""
    print(f"\nEvaluating {dataset_name}")

    # Fast Fractal Flow Network
    ffn = FastFractalFlowNetwork(max_depth=3, threshold=0.1)
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