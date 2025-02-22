import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
import time
from sklearn.cluster import KMeans

# Additional classifiers from scikit-learn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# --- Custom FractalFlowNet (FFN) --- #
class FFNNode:
    def __init__(self, center, radius, value=None, fractal_dim=0.0):
        self.center = center
        self.radius = radius
        self.children = []
        self.value = value
        self.fractal_dim = fractal_dim  # Local fractal dimension estimate
        self.error_count = 0          # Track local mispredictions for adaptation

    def is_leaf(self):
        return len(self.children) == 0 and self.value is not None


class FractalFlowNet:
    def __init__(self, max_depth=5, min_samples_split=5, max_splits=4,
                 adapt_threshold=0.1, momentum=0.9, flow_temp=1.0):
        """
        Parameters:
         - max_depth: Maximum recursion depth for network growth.
         - min_samples_split: Minimum number of samples to attempt a split.
         - max_splits: Maximum number of child nodes a node can split into.
         - adapt_threshold: Error ratio threshold to trigger local adaptation.
         - momentum: Controls how much a node’s center is updated during adaptation.
         - flow_temp: Temperature parameter controlling the softness of flow weights.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_splits = max_splits
        self.adapt_threshold = adapt_threshold
        self.momentum = momentum
        self.flow_temp = flow_temp
        self.root = None

    def fit(self, X, y):
        # Initialize with a single seed node (global mean and radius)
        center = np.mean(X, axis=0)
        radius = np.max(np.linalg.norm(X - center, axis=1))
        self.root = FFNNode(center, radius)
        self._grow_network(X, y, self.root, depth=0)

    def _entropy(self, y):
        hist = np.bincount(y, minlength=np.max(y) + 1)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _estimate_fractal_dim(self, X):
        """
        Estimate the fractal dimension using a correlation dimension approximation.
        """
        if len(X) < 2:
            return 1.0
        distances = pdist(X)
        if np.max(distances) == 0:
            return 1.0
        r_values = np.percentile(distances, np.linspace(10, 90, 9)) + 1e-10
        C_r = [np.mean(distances < r) for r in r_values]
        C_r = np.array(C_r)
        C_r = np.maximum(C_r, 1e-10)
        log_r = np.log(r_values)
        log_C_r = np.log(C_r)
        slope, _ = np.polyfit(log_r, log_C_r, 1)
        return slope

    def _decide_splits(self, y, X):
        unique_classes = len(np.unique(y))
        if unique_classes <= 1:
            return 1
        max_entropy = np.log2(unique_classes)
        entropy = self._entropy(y)
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
        fractal_dim = self._estimate_fractal_dim(X)
        complexity = norm_entropy * (fractal_dim / X.shape[1])
        if complexity < 0.1:
            return 1
        elif complexity < 0.4:
            return 2
        elif complexity < 0.7:
            return 3
        else:
            return min(self.max_splits, int(np.ceil(fractal_dim)))

    def _grow_network(self, X, y, node, depth):
        if len(y) < self.min_samples_split or depth >= self.max_depth:
            node.value = Counter(y).most_common(1)[0][0] if len(y) > 0 else 0
            return

        k = self._decide_splits(y, X)
        node.fractal_dim = self._estimate_fractal_dim(X)
        if k == 1:
            node.value = Counter(y).most_common(1)[0][0]
            return

        # Optionally reduce dimensionality via PCA for clustering efficiency
        if X.shape[0] > k and X.shape[1] > 1:
            pca = PCA(n_components=min(k - 1, X.shape[1]))
            X_proj = pca.fit_transform(X)
        else:
            X_proj = X

        kmeans = KMeans(n_clusters=k, random_state=0).fit(X_proj)
        labels = kmeans.labels_
        child_radius = node.radius / np.sqrt(k)

        for i in range(k):
            cluster_idx = labels == i
            if np.sum(cluster_idx) == 0:
                continue
            child_data = X[cluster_idx]
            child_y = y[cluster_idx]
            child_center = np.mean(child_data, axis=0)
            child_node = FFNNode(child_center, child_radius)
            node.children.append(child_node)
            self._grow_network(child_data, child_y, child_node, depth + 1)

        if not node.children:
            node.value = Counter(y).most_common(1)[0][0]

    def _compute_flow_weights(self, x, children):
        # Compute distances from x to each child's center
        distances = np.array([np.linalg.norm(x - child.center) for child in children])
        # Softmax weighting with flow temperature
        weights = np.exp(-distances / self.flow_temp)
        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
        else:
            weights = weights / np.sum(weights)
        return weights

    def _traverse_network(self, x, node, adapt=False, true_label=None):
        if node.is_leaf():
            if adapt and true_label is not None and node.value != true_label:
                node.error_count += 1
                if node.error_count / (len(node.children) + 1) > self.adapt_threshold:
                    self._adapt_node(node, x, true_label)
            return node.value

        # Compute flow weights and choose the child with the highest weight
        weights = self._compute_flow_weights(x, node.children)
        chosen_idx = np.argmax(weights)
        chosen_child = node.children[chosen_idx]

        # Check for adaptation: if prediction from chosen branch is off, perform liquid splitting
        if adapt and true_label is not None:
            pred = self._traverse_network(x, chosen_child, adapt=False)
            if pred != true_label:
                node.error_count += 1
                if node.error_count / (len(node.children) + 1) > self.adapt_threshold:
                    self._liquid_split(node, x, true_label)
        return self._traverse_network(x, chosen_child, adapt=adapt, true_label=true_label)

    def predict(self, X, y=None, adapt=False):
        preds = []
        for i, x in enumerate(X):
            pred = self._traverse_network(x, self.root, adapt=adapt,
                                          true_label=y[i] if y is not None else None)
            preds.append(pred)
        return np.array(preds)

    def _liquid_split(self, node, x, true_label):
        # Create a new branch if maximum splits not reached
        if len(node.children) >= self.max_splits:
            return
        new_center = x
        new_radius = node.radius / 2
        new_node = FFNNode(new_center, new_radius, value=true_label)
        node.children.append(new_node)
        node.error_count = 0

    def _adapt_node(self, node, x, true_label):
        # Update the node’s center using momentum and adjust its label to the true label
        node.center = self.momentum * node.center + (1 - self.momentum) * x
        node.value = true_label
        node.error_count = 0


# --- Evaluation Function --- #
def evaluate_models(dataset_name):
    # Load dataset
    if dataset_name == "Iris":
        dataset = load_iris()
    elif dataset_name == "Wine":
        dataset = load_wine()
    elif dataset_name == "Breast Cancer":
        dataset = load_breast_cancer()
    else:
        raise ValueError("Unknown dataset")

    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {}
    results = {}

    # --- 1. Custom FractalFlowNet (FFN) --- #
    ffn = FractalFlowNet(max_depth=5, min_samples_split=5, max_splits=4,
                         adapt_threshold=0.1, momentum=0.9, flow_temp=1.0)
    start = time.time()
    ffn.fit(X_train, y_train)
    train_time = time.time() - start
    y_pred = ffn.predict(X_test, y_test, adapt=True)
    acc = accuracy_score(y_test, y_pred)
    models["FFN"] = ffn
    results["FFN"] = {"train_time": train_time, "accuracy": acc}

    # --- 2. Decision Tree --- #
    dt = DecisionTreeClassifier(random_state=42)
    start = time.time()
    dt.fit(X_train, y_train)
    train_time = time.time() - start
    y_pred = dt.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    models["Decision Tree"] = dt
    results["Decision Tree"] = {"train_time": train_time, "accuracy": acc}

    # --- 3. Random Forest --- #
    rf = RandomForestClassifier(random_state=42)
    start = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - start
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    models["Random Forest"] = rf
    results["Random Forest"] = {"train_time": train_time, "accuracy": acc}

    # --- 4. Logistic Regression --- #
    lr = LogisticRegression(max_iter=1000, random_state=42)
    start = time.time()
    lr.fit(X_train, y_train)
    train_time = time.time() - start
    y_pred = lr.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    models["Logistic Regression"] = lr
    results["Logistic Regression"] = {"train_time": train_time, "accuracy": acc}

    # --- 5. Support Vector Classifier (SVC) --- #
    svc = SVC(random_state=42)
    start = time.time()
    svc.fit(X_train, y_train)
    train_time = time.time() - start
    y_pred = svc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    models["SVC"] = svc
    results["SVC"] = {"train_time": train_time, "accuracy": acc}

    # Print side-by-side results
    print(f"\n--- {dataset_name} Dataset Comparison ---")
    print("{:<20} {:<15} {:<15}".format("Model", "Train Time (s)", "Accuracy"))
    for model_name, metrics in results.items():
        print("{:<20} {:<15.4f} {:<15.4f}".format(model_name, metrics["train_time"], metrics["accuracy"]))


if __name__ == "__main__":
    for dataset in ["Iris", "Wine", "Breast Cancer"]:
        evaluate_models(dataset)
