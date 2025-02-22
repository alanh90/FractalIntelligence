import numpy as np
from collections import Counter
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time

# Additional classifiers from scikit-learn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# --- Mandala Fractal Node and Network --- #
class MandalaFractalNode:
    def __init__(self, center, radius, value=None):
        """
        Represents a 'box' that encodes a region of the latent space.
        - center: The numeric center (abstract encoding) of the box.
        - radius: A measure of the box's size.
        - value: The predicted label if this node is a leaf.
        """
        self.center = center
        self.radius = radius
        self.value = value
        self.children = []  # Sub-boxes arranged in a mandala (radial) pattern.
        self.error_count = 0  # Count mispredictions for local adaptation.

    def is_leaf(self):
        return len(self.children) == 0 and self.value is not None


class MandalaFractalNet:
    def __init__(self, max_depth=5, min_samples_split=5, max_children=4,
                 adapt_threshold=0.1, momentum=0.9, flow_temp=1.0):
        """
        Parameters:
          - max_depth: Maximum recursive depth for growing the network.
          - min_samples_split: Minimum samples required to split a node.
          - max_children: Maximum number of child boxes (a mandala pattern).
          - adapt_threshold: Local error ratio threshold for triggering adjustments.
          - momentum: How much a node's center is updated (a fractal adjuster).
          - flow_temp: Temperature controlling how 'flow' is computed during prediction.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_children = max_children
        self.adapt_threshold = adapt_threshold
        self.momentum = momentum
        self.flow_temp = flow_temp
        self.root = None

    def fit(self, X, y):
        # Initialize with a global box covering all data.
        center = np.mean(X, axis=0)
        radius = np.max(np.linalg.norm(X - center, axis=1))
        self.root = MandalaFractalNode(center, radius)
        self._grow_network(X, y, self.root, depth=0)

    def _grow_network(self, X, y, node, depth):
        # If too few samples, at max depth, or data is pure, mark as leaf.
        if len(y) < self.min_samples_split or depth >= self.max_depth or len(np.unique(y)) == 1:
            node.value = Counter(y).most_common(1)[0][0] if len(y) > 0 else 0
            return

        # Create children arranged in a mandala (radial, symmetric) pattern.
        d = node.center.shape[0]
        children = []
        for i in range(self.max_children):
            # Generate a random unit vector in the latent space.
            direction = np.random.randn(d)
            direction /= np.linalg.norm(direction)
            # "Box" idea: offset is a fraction of parent's radius.
            offset = (node.radius / 2) * direction
            child_center = node.center + offset
            child_radius = node.radius / 2
            child = MandalaFractalNode(child_center, child_radius)
            children.append(child)

        # Vectorized assignment of points to the nearest child center.
        centers = np.array([child.center for child in children])
        distances = np.linalg.norm(X[:, None] - centers[None, :], axis=2)  # shape: (n_samples, max_children)
        chosen = np.argmin(distances, axis=1)
        assignments = [X[chosen == i] for i in range(self.max_children)]
        y = np.array(y)
        indices = [np.where(chosen == i)[0] for i in range(self.max_children)]

        # Recursively grow each child that has assigned points.
        for i, child in enumerate(children):
            if assignments[i].shape[0] > 0:
                self._grow_network(assignments[i], y[indices[i]], child, depth + 1)
                node.children.append(child)

        # If no child was created, mark this node as a leaf.
        if len(node.children) == 0:
            node.value = Counter(y).most_common(1)[0][0]

    def _compute_flow_weights(self, x, children):
        # Vectorized computation: stack child centers and compute distances.
        centers = np.array([child.center for child in children])
        distances = np.linalg.norm(centers - x, axis=1)
        weights = np.exp(-distances / self.flow_temp)
        total = np.sum(weights)
        if total == 0:
            weights = np.ones_like(weights)
        else:
            weights = weights / total
        return weights

    def _traverse(self, x, node, adapt=False, true_label=None):
        if node.is_leaf():
            if adapt and true_label is not None and node.value != true_label:
                node.error_count += 1
                if node.error_count > self.adapt_threshold * (len(node.children) + 1):
                    self._adapt_node(node, x, true_label)
            return node.value

        # Compute flow weights and choose a child.
        weights = self._compute_flow_weights(x, node.children)
        chosen_index = np.argmax(weights)
        chosen_child = node.children[chosen_index]

        # Adaptation: if the chosen branch misclassifies, trigger a "liquid split".
        if adapt and true_label is not None:
            pred = self._traverse(x, chosen_child, adapt=False)
            if pred != true_label:
                node.error_count += 1
                if node.error_count > self.adapt_threshold * (len(node.children) + 1):
                    self._liquid_split(node, x, true_label)
        return self._traverse(x, chosen_child, adapt=adapt, true_label=true_label)

    def predict(self, X, y=None, adapt=False):
        preds = []
        for i, x in enumerate(X):
            pred = self._traverse(x, self.root, adapt=adapt, true_label=y[i] if y is not None else None)
            preds.append(pred)
        return np.array(preds)

    def _liquid_split(self, node, x, true_label):
        # If the node hasn't reached its maximum children, add a new branch.
        if len(node.children) >= self.max_children:
            return
        new_center = x
        new_radius = node.radius / 2
        new_node = MandalaFractalNode(new_center, new_radius, value=true_label)
        node.children.append(new_node)
        node.error_count = 0

    def _adapt_node(self, node, x, true_label):
        # Adjust the node's center (the "box adjuster") using momentum.
        node.center = self.momentum * node.center + (1 - self.momentum) * x
        node.value = true_label
        node.error_count = 0


# --- Evaluation Function --- #
def evaluate_models(dataset_name):
    # Load the dataset.
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

    # Standardize features.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {}
    results = {}

    # --- 1. Custom MandalaFractalNet --- #
    mandala_net = MandalaFractalNet(max_depth=5, min_samples_split=5, max_children=4,
                                    adapt_threshold=0.1, momentum=0.9, flow_temp=1.0)
    start = time.time()
    mandala_net.fit(X_train, y_train)
    train_time = time.time() - start
    y_pred = mandala_net.predict(X_test, y_test, adapt=True)
    acc = accuracy_score(y_test, y_pred)
    models["MandalaFractalNet"] = mandala_net
    results["MandalaFractalNet"] = {"train_time": train_time, "accuracy": acc}

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

    # --- 6. MLP Neural Network --- #
    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
    start = time.time()
    mlp.fit(X_train, y_train)
    train_time = time.time() - start
    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    models["MLPClassifier"] = mlp
    results["MLPClassifier"] = {"train_time": train_time, "accuracy": acc}

    # --- 7. K-Nearest Neighbors --- #
    knn = KNeighborsClassifier()
    start = time.time()
    knn.fit(X_train, y_train)
    train_time = time.time() - start
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    models["KNN"] = knn
    results["KNN"] = {"train_time": train_time, "accuracy": acc}

    # Print side-by-side results.
    print(f"\n--- {dataset_name} Dataset Comparison ---")
    print("{:<25} {:<15} {:<15}".format("Model", "Train Time (s)", "Accuracy"))
    for model_name, metrics in results.items():
        print("{:<25} {:<15.4f} {:<15.4f}".format(model_name, metrics["train_time"], metrics["accuracy"]))


if __name__ == "__main__":
    for dataset in ["Iris", "Wine", "Breast Cancer"]:
        evaluate_models(dataset)
