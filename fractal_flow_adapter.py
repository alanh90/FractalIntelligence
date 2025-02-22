import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_counts_split import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time

# FFNNode class with evolution capabilities
class FFNNode:
    def __init__(self, center, radius, value=None, fractal_dim=0.0):
        self.center = center
        self.radius = radius
        self.children = []
        self.value = value
        self.fractal_dim = fractal_dim  # Local fractal dimension estimate
        self.error_count = 0  # Track mispredictions for feedback

    def is_leaf(self):
        return len(self.children) == 0 and self.value is not None

# Enhanced FractalFlowNetwork with liquid evolution and fractal feedback
class FractalFlowNetwork:
    def __init__(self, max_depth=5, min_samples_split=5, max_splits=4, adapt_threshold=0.1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_splits = max_splits
        self.adapt_threshold = adapt_threshold  # Threshold for inference-time adaptation
        self.root = None

    def fit(self, X, y):
        center = np.mean(X, axis=0)
        radius = np.max(np.linalg.norm(X - center, axis=1))
        self.root = FFNNode(center, radius)
        self._grow_network(X, y, self.root, depth=0)

    def _entropy(self, y):
        hist = np.bincount(y, minlength=np.max(y) + 1)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _estimate_fractal_dim(self, X):
        """Simple box-counting proxy using variance across dimensions."""
        if len(X) < 2:
            return 1.0
        scales = [0.1, 0.5, 1.0]  # Different grid sizes
        counts = []
        for scale in scales:
            bins = np.ceil((np.max(X, axis=0) - np.min(X, axis=0)) / scale).astype(int)
            hist, _ = np.histogramdd(X, bins=bins)
            counts.append(np.sum(hist > 0))
        counts = np.log(counts + 1)  # Avoid log(0)
        scales = np.log(1 / np.array(scales))
        if np.std(counts) == 0:
            return 1.0
        return np.polyfit(scales, counts, 1)[0]  # Slope as fractal dim estimate

    def _decide_splits(self, y, X):
        """Decide splits based on entropy and fractal dimension."""
        unique_classes = len(np.unique(y))
        if unique_classes <= 1:
            return 1
        max_entropy = np.log2(unique_classes)
        entropy = self._entropy(y)
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
        fractal_dim = self._estimate_fractal_dim(X)

        # Combine entropy and fractal dimension
        complexity = norm_entropy * (fractal_dim / X.shape[1])  # Normalize by dims
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

        # PCA and k-means for splitting
        if X.shape[0] > k and X.shape[1] > 1:
            pca = PCA(n_components=min(k-1, X.shape[1]))
            X_proj = pca.fit_transform(X)
        else:
            X_proj = X

        kmeans = KMeans(n_clusters=k, random_state=0).fit(X_proj)
        labels = kmeans.labels_

        child_radius = node.radius / np.sqrt(k)  # Adjust radius dynamically
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

    def predict(self, X, y=None, adapt=False):
        """Predict with optional adaptation based on feedback."""
        preds = []
        for i, x in enumerate(X):
            pred = self._traverse_network(x, self.root, adapt=adapt, true_label=y[i] if y is not None else None)
            preds.append(pred)
        return np.array(preds)

    def _traverse_network(self, x, node, adapt=False, true_label=None):
        if node.is_leaf():
            if adapt and true_label is not None and node.value != true_label:
                node.error_count += 1
                if node.error_count / 10 > self.adapt_threshold:  # Trigger adaptation
                    self._adapt_node(node, x, true_label)
            return node.value

        distances = [np.linalg.norm(x - child.center) for child in node.children]
        closest_idx = np.argmin(distances)
        closest_child = node.children[closest_idx]

        if adapt and true_label is not None:
            pred = self._traverse_network(x, closest_child, adapt=False)
            if pred != true_label:
                node.error_count += 1
                if node.error_count / 10 > self.adapt_threshold:
                    self._liquid_split(node, x, true_label)
        return self._traverse_network(x, closest_child, adapt=adapt, true_label=true_label)

    def _liquid_split(self, node, x, true_label):
        """Add a new child node dynamically during inference."""
        if len(node.children) >= self.max_splits:
            return
        new_center = x  # Position new node at the misclassified point
        new_radius = node.radius / 2
        new_node = FFNNode(new_center, new_radius, value=true_label)
        node.children.append(new_node)
        node.error_count = 0  # Reset error count

    def _adapt_node(self, node, x, true_label):
        """Adjust leaf node based on feedback."""
        node.center = 0.9 * node.center + 0.1 * x  # Gradual shift toward new data
        node.value = true_label  # Update value
        node.error_count = 0

# Evaluate the enhanced FFN
def evaluate_ffn(dataset_name):
    dataset_loaders = {
        "Iris": load_iris,
        "Wine": load_wine,
        "Breast Cancer": load_breast_cancer
    }
    dataset = dataset_loaders[dataset_name]()
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    ffn = FractalFlowNetwork(max_depth=5, min_samples_split=5, max_splits=4, adapt_threshold=0.1)
    start_time = time.time()
    ffn.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Predict with adaptation
    y_pred = ffn.predict(X_test, y_test, adapt=True)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n--- {dataset_name} Dataset ---")
    print(f"FFN Training Time: {train_time:.4f} seconds")
    print(f"FFN Accuracy with Adaptation: {accuracy:.4f}")

if __name__ == "__main__":
    for dataset in ["Iris", "Wine", "Breast Cancer"]:
        evaluate_ffn(dataset)