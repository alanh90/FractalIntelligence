import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import time

# FFNNode class
class FFNNode:
    def __init__(self, center, radius, value=None):
        self.center = center
        self.radius = radius
        self.children = []
        self.value = value

    def is_leaf(self):
        return len(self.children) == 0 and self.value is not None

# FractalFlowNetwork class
class FractalFlowNetwork:
    def __init__(self, max_depth=5, min_samples_split=5, n_splits=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_splits = n_splits
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

    def _grow_network(self, X, y, node, depth):
        if (len(y) < self.min_samples_split or
            depth >= self.max_depth or
            self._entropy(y) < 0.1):
            node.value = Counter(y).most_common(1)[0][0]
            return

        if X.shape[0] > 1 and X.shape[1] > 1:
            pca = PCA(n_components=1)
            pca.fit(X)
            direction = pca.components_[0]
        else:
            direction = np.random.randn(X.shape[1])
            direction /= np.linalg.norm(direction)

        projections = np.dot(X - node.center, direction)
        median_proj = np.median(projections)

        left_idx = projections <= median_proj
        right_idx = projections > median_proj

        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            node.value = Counter(y).most_common(1)[0][0]
            return

        left_center = node.center + direction * (median_proj - node.radius)
        right_center = node.center + direction * (median_proj + node.radius)
        child_radius = node.radius / 2

        left_child = FFNNode(left_center, child_radius)
        right_child = FFNNode(right_center, child_radius)
        node.children = [left_child, right_child]

        self._grow_network(X[left_idx], y[left_idx], left_child, depth + 1)
        self._grow_network(X[right_idx], y[right_idx], right_child, depth + 1)

        for child in node.children:
            if child.children:
                child_data = X[np.linalg.norm(X - child.center, axis=1) <= child.radius]
                if len(child_data) > 0:
                    child.center = np.mean(child_data, axis=0)

    def predict(self, X):
        return np.array([self._traverse_network(x, self.root) for x in X])

    def _traverse_network(self, x, node):
        if node.is_leaf():
            return node.value
        distances = [np.linalg.norm(x - child.center) for child in node.children]
        closest_child = node.children[np.argmin(distances)]
        return self._traverse_network(x, closest_child)

# Function to build a simple neural network
def build_neural_network(input_shape, num_classes):
    model = Sequential([
        Dense(16, activation='relu', input_shape=(input_shape,)),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to evaluate a model on a dataset
def evaluate_model(model, X_train, y_train, X_test, y_test, is_ffn=True):
    start_time = time.time()
    if is_ffn:
        model.fit(X_train, y_train)
    else:
        y_train_cat = to_categorical(y_train)
        model.fit(X_train, y_train_cat, epochs=50, verbose=0)
    train_time = time.time() - start_time

    if is_ffn:
        y_pred = model.predict(X_test)
    else:
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    accuracy = accuracy_score(y_test, y_pred)

    return train_time, accuracy

# Main function to compare FFN and NN on multiple datasets
def compare_on_datasets():
    datasets = {
        "Iris": load_iris(),
        "Wine": load_wine(),
        "Breast Cancer": load_breast_cancer()
    }

    for name, dataset in datasets.items():
        print(f"\n--- {name} Dataset ---")
        X, y = dataset.data, dataset.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train and evaluate FFN
        ffn = FractalFlowNetwork(max_depth=5, min_samples_split=5, n_splits=2)
        ffn_time, ffn_accuracy = evaluate_model(ffn, X_train, y_train, X_test, y_test, is_ffn=True)

        # Train and evaluate Neural Network
        num_classes = len(np.unique(y))
        nn_model = build_neural_network(X_train.shape[1], num_classes)
        nn_time, nn_accuracy = evaluate_model(nn_model, X_train, y_train, X_test, y_test, is_ffn=False)

        # Print results
        print(f"FFN Training Time: {ffn_time:.4f} seconds")
        print(f"FFN Accuracy: {ffn_accuracy:.4f}")
        print(f"Neural Network Training Time: {nn_time:.4f} seconds")
        print(f"Neural Network Accuracy: {nn_accuracy:.4f}")

if __name__ == "__main__":
    compare_on_datasets()