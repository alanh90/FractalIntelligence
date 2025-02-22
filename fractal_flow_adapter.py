import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Node class for the Fractal Flow Network
class FFNNode:
    def __init__(self, center, radius, value=None):
        self.center = center
        self.radius = radius
        self.children = []
        self.value = value

    def is_leaf(self):
        return len(self.children) == 0 and self.value is not None

# Improved Fractal Flow Network class
class FractalFlowNetwork:
    def __init__(self, max_depth=5, min_samples_split=5, n_splits=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_splits = n_splits  # Limited to 2 for simplicity, can be adaptive later
        self.root = None

    def fit(self, X, y):
        # Initialize root node with data centroid and max radius
        center = np.mean(X, axis=0)
        radius = np.max(np.linalg.norm(X - center, axis=1))
        self.root = FFNNode(center, radius)
        self._grow_network(X, y, self.root, depth=0)

    def _entropy(self, y):
        # Calculate entropy to assess class purity
        hist = np.bincount(y, minlength=np.max(y) + 1)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _grow_network(self, X, y, node, depth):
        # Stop if too few samples, max depth reached, or low entropy
        if (len(y) < self.min_samples_split or
            depth >= self.max_depth or
            self._entropy(y) < 0.1):
            node.value = Counter(y).most_common(1)[0][0]
            return

        # Use PCA to find the principal direction for splitting
        if X.shape[0] > 1 and X.shape[1] > 1:
            pca = PCA(n_components=1)
            pca.fit(X)
            direction = pca.components_[0]
        else:
            direction = np.random.randn(X.shape[1])
            direction /= np.linalg.norm(direction)

        # Project data onto the principal direction
        projections = np.dot(X - node.center, direction)
        median_proj = np.median(projections)

        # Split data at the median
        left_idx = projections <= median_proj
        right_idx = projections > median_proj

        # If split is imbalanced, assign a leaf value
        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            node.value = Counter(y).most_common(1)[0][0]
            return

        # Create child nodes with adjusted centers and halved radius
        left_center = node.center + direction * (median_proj - node.radius)
        right_center = node.center + direction * (median_proj + node.radius)
        child_radius = node.radius / 2

        left_child = FFNNode(left_center, child_radius)
        right_child = FFNNode(right_center, child_radius)
        node.children = [left_child, right_child]

        # Recursively grow children
        self._grow_network(X[left_idx], y[left_idx], left_child, depth + 1)
        self._grow_network(X[right_idx], y[right_idx], right_child, depth + 1)

        # Refine child centers to "flow" toward data centroids
        for child in node.children:
            if child.children:  # Only refine if not a leaf
                child_data = X[np.linalg.norm(X - child.center, axis=1) <= child.radius]
                if len(child_data) > 0:
                    child.center = np.mean(child_data, axis=0)

    def predict(self, X):
        return np.array([self._traverse_network(x, self.root) for x in X])

    def _traverse_network(self, x, node):
        if node.is_leaf():
            return node.value
        # Assign to closest child based on distance
        distances = [np.linalg.norm(x - child.center) for child in node.children]
        closest_child = node.children[np.argmin(distances)]
        return self._traverse_network(x, closest_child)

# Test the improved FFN on the Iris dataset
def test_speed_and_accuracy():
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Improved Fractal Flow Network (FFN)...")
    start_time = time.time()
    ffn = FractalFlowNetwork(max_depth=5, min_samples_split=5, n_splits=2)
    ffn.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Predict and evaluate
    y_pred = ffn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nResults:")
    print(f"FFN Training Time: {train_time:.4f} seconds")
    print(f"FFN Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    test_speed_and_accuracy()