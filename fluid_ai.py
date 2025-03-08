import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import time
from scipy import stats
from scipy.spatial import distance
import random
import math
from collections import defaultdict, Counter


# --- Fluid Topology Self-Adjusting Network --- #
class FluidTopoNetwork(BaseEstimator, ClassifierMixin):
    def __init__(self, turbulence=0.1, flow_rate=0.05, viscosity=0.8, pressure_sensitivity=1.0):
        """
        Fluid Topology Network that dynamically adjusts its structure.

        Parameters:
        - turbulence: Amount of randomness in the flow dynamics
        - flow_rate: Speed of information flow through the network
        - viscosity: Resistance to structural changes (higher = more stable)
        - pressure_sensitivity: Sensitivity to input data pressure points
        """
        self.turbulence = turbulence
        self.flow_rate = flow_rate
        self.viscosity = viscosity
        self.pressure_sensitivity = pressure_sensitivity

        # Internal state
        self.flow_pathways = None
        self.pressure_points = None
        self.vortices = None
        self.class_mappings = None

    def _initialize_fluid_state(self, X, y):
        """Initialize the fluid network state based on data."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Create initial pressure points (data concentration areas)
        n_points = min(n_samples // 5, 20)
        indices = np.random.choice(n_samples, size=n_points, replace=False)
        self.pressure_points = X[indices].copy()

        # Initialize flow pathways (connection strengths between pressure points)
        self.flow_pathways = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(i + 1, n_points):
                # Initial flow strength based on distance
                dist = np.linalg.norm(self.pressure_points[i] - self.pressure_points[j])
                flow = 1.0 / (1.0 + dist)
                self.flow_pathways[i, j] = flow
                self.flow_pathways[j, i] = flow

        # Initialize vortices (classification centers)
        self.vortices = []
        for class_idx in range(n_classes):
            class_samples = X[y == class_idx]
            if len(class_samples) > 0:
                # Create vortex at class center
                vortex_center = np.mean(class_samples, axis=0)
                vortex_strength = len(class_samples) / n_samples
                self.vortices.append((vortex_center, vortex_strength, class_idx))

        # Initialize class mappings
        self.class_mappings = {i: i for i in range(n_classes)}

    def _fluid_dynamics(self, X):
        """Simulate fluid dynamics through the network."""
        n_samples = X.shape[0]
        n_points = len(self.pressure_points)

        # Calculate pressure at each point
        pressures = np.zeros(n_points)
        for i, point in enumerate(self.pressure_points):
            # Calculate distance-based pressure from each sample
            distances = np.linalg.norm(X - point, axis=1)
            # Pressure decreases with distance (inverse square law)
            point_pressure = np.sum(1.0 / (1.0 + distances ** 2))
            pressures[i] = point_pressure * self.pressure_sensitivity

        # Update flow pathways based on pressure differences
        for i in range(n_points):
            for j in range(i + 1, n_points):
                # Flow from high to low pressure
                pressure_diff = abs(pressures[i] - pressures[j])
                flow_direction = 1 if pressures[i] > pressures[j] else -1

                # Flow update with viscosity damping
                current_flow = self.flow_pathways[i, j]
                new_flow = current_flow + flow_direction * pressure_diff * self.flow_rate
                new_flow = self.viscosity * current_flow + (1 - self.viscosity) * new_flow

                # Add turbulence
                turbulence_factor = np.random.normal(0, self.turbulence)
                new_flow += turbulence_factor

                # Update flow (keep symmetric)
                self.flow_pathways[i, j] = max(0, new_flow)
                self.flow_pathways[j, i] = self.flow_pathways[i, j]

        # Adjust pressure points based on data flow
        movement = np.zeros_like(self.pressure_points)
        for i, point in enumerate(self.pressure_points):
            # Calculate weighted center from connected points
            for j, other_point in enumerate(self.pressure_points):
                if i != j:
                    flow = self.flow_pathways[i, j]
                    direction = other_point - point
                    movement[i] += flow * direction / (np.linalg.norm(direction) + 1e-10)

            # Normalize movement by total flow
            total_flow = np.sum(self.flow_pathways[i])
            if total_flow > 0:
                movement[i] /= total_flow

        # Apply movement with viscosity damping
        self.pressure_points += movement * (1 - self.viscosity) * self.flow_rate

        return pressures

    def fit(self, X, y):
        """Fit the fluid network to training data."""
        if X.size == 0 or y.size == 0:
            raise ValueError("Empty input data provided.")

        # Initialize fluid state
        self._initialize_fluid_state(X, y)

        # Run fluid dynamics for multiple iterations
        n_iterations = 10
        for _ in range(n_iterations):
            # Update fluid dynamics
            self._fluid_dynamics(X)

            # Update vortices based on class data
            new_vortices = []
            for class_idx in np.unique(y):
                class_samples = X[y == class_idx]
                if len(class_samples) > 0:
                    # Get pressure points most associated with this class
                    class_pressures = np.zeros(len(self.pressure_points))
                    for i, point in enumerate(self.pressure_points):
                        class_distances = np.linalg.norm(class_samples - point, axis=1)
                        class_pressures[i] = np.sum(1.0 / (1.0 + class_distances ** 2))

                    # Create new vortex with influence from top pressure points
                    top_indices = np.argsort(class_pressures)[-3:]
                    vortex_center = np.mean(self.pressure_points[top_indices], axis=0)
                    vortex_strength = len(class_samples) / len(X)
                    new_vortices.append((vortex_center, vortex_strength, class_idx))

            self.vortices = new_vortices

        return self

    def predict(self, X):
        """Predict class labels using fluid dynamics."""
        if X.size == 0:
            raise ValueError("Empty input data for prediction.")

        predictions = []

        for x in X:
            # Calculate influence from each vortex
            vortex_influences = []
            for center, strength, class_idx in self.vortices:
                # Distance-based influence with vortex strength
                distance = np.linalg.norm(x - center)
                influence = strength / (1.0 + distance ** 2)
                vortex_influences.append((influence, class_idx))

            # Find vortex with strongest influence
            if vortex_influences:
                predicted_class = max(vortex_influences, key=lambda x: x[0])[1]
            else:
                predicted_class = 0  # Default if no vortices

            predictions.append(predicted_class)

        return np.array(predictions)


# --- Enhanced FluidTopoNetwork --- #
class EnhancedFluidTopoNetwork(FluidTopoNetwork):
    def __init__(self, turbulence=0.1, flow_rate=0.05, viscosity=0.8,
                 pressure_sensitivity=1.0, vortex_interaction=0.2, adaptive_turbulence=True):
        super().__init__(turbulence, flow_rate, viscosity, pressure_sensitivity)
        self.vortex_interaction = vortex_interaction
        self.adaptive_turbulence = adaptive_turbulence
        self.error_history = []

    def _vortex_interactions(self):
        """Simulate interactions between vortices (creates more complex dynamics)."""
        if not self.vortices or len(self.vortices) < 2:
            return

        new_vortices = []
        for i, (center_i, strength_i, class_i) in enumerate(self.vortices):
            # Calculate interactions with other vortices
            net_movement = np.zeros_like(center_i)

            for j, (center_j, strength_j, class_j) in enumerate(self.vortices):
                if i == j:
                    continue

                # Calculate direction and distance
                direction = center_j - center_i
                distance = np.linalg.norm(direction)

                if distance > 0:
                    # Normalize direction
                    direction = direction / distance

                    # Calculate interaction force (repulsive for same class, attractive for different)
                    force = self.vortex_interaction * strength_j
                    if class_i == class_j:
                        # Repulsive force between same-class vortices (creates better spacing)
                        force = -force / (distance + 0.1)
                    else:
                        # Attractive force for different classes but weaker with distance
                        force = force / (distance ** 2 + 1.0)

                    # Apply force as movement
                    net_movement += direction * force

            # Calculate new center with capped movement
            movement_magnitude = np.linalg.norm(net_movement)
            if movement_magnitude > 0.5:  # Cap maximum movement
                net_movement = net_movement * 0.5 / movement_magnitude

            new_center = center_i + net_movement
            new_vortices.append((new_center, strength_i, class_i))

        self.vortices = new_vortices

    def _adaptive_parameters(self, X, y=None):
        """Adapt parameters based on data and performance."""
        if not self.adaptive_turbulence or y is None:
            return

        # Check recent error history to adapt turbulence
        if self.error_history:
            # If error is increasing, increase turbulence for exploration
            if len(self.error_history) >= 2 and self.error_history[-1] > self.error_history[-2]:
                self.turbulence = min(0.3, self.turbulence * 1.2)
            else:
                # If error is decreasing, reduce turbulence for exploitation
                self.turbulence = max(0.01, self.turbulence * 0.9)

    def fit(self, X, y):
        """Enhanced fit method with vortex interactions."""
        super().fit(X, y)

        # Additional iterations with vortex interactions
        for _ in range(5):
            self._vortex_interactions()
            self._fluid_dynamics(X)
            self._adaptive_parameters(X, y)

            # Validate current model
            if len(X) > 10:
                preds = self.predict(X)
                error = np.mean(preds != y)
                self.error_history.append(error)

        return self

    def predict(self, X):
        """Enhanced prediction with uncertainty handling."""
        predictions = super().predict(X)

        # For samples near vortex boundaries, handle uncertainty
        uncertainties = []
        for x in X:
            # Calculate influence from each vortex
            vortex_influences = []
            for center, strength, class_idx in self.vortices:
                distance = np.linalg.norm(x - center)
                influence = strength / (1.0 + distance ** 2)
                vortex_influences.append((influence, class_idx))

            # Sort influences
            vortex_influences.sort(reverse=True, key=lambda x: x[0])

            # Check if top two influences are close (uncertainty)
            if len(vortex_influences) >= 2:
                top1, top2 = vortex_influences[0], vortex_influences[1]
                uncertainty = 1.0 - (top1[0] - top2[0]) / (top1[0] + 1e-10)
            else:
                uncertainty = 0.0

            uncertainties.append(uncertainty)

        # For high uncertainty points, check nearby samples for consensus
        for i, uncertainty in enumerate(uncertainties):
            if uncertainty > 0.4:  # High uncertainty threshold
                # Current prediction might be unreliable
                # Keep it as is for now, but this could be enhanced
                pass

        return predictions


# --- Dataset Loader --- #
def load_dataset(name):
    """Load and preprocess a dataset."""
    try:
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
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {name}: {str(e)}")


# --- Model Evaluation --- #
def evaluate_models(dataset_name, X_train, X_test, y_train, y_test):
    """Evaluate fluid network models on a dataset."""
    print(f"\nEvaluating {dataset_name}")

    models = [
        ("FluidTopoNetwork", FluidTopoNetwork(turbulence=0.1, flow_rate=0.05)),
        ("EnhancedFluidTopoNetwork", EnhancedFluidTopoNetwork(turbulence=0.1, flow_rate=0.05, vortex_interaction=0.2))
    ]

    for name, model in models:
        try:
            start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"{name} - Time: {train_time:.4f}s, Accuracy: {acc:.4f}")
        except Exception as e:
            print(f"{name} - Error: {str(e)}")


# --- Main Execution --- #
if __name__ == "__main__":
    datasets = ["Iris", "Wine", "Breast Cancer"]
    for dataset in datasets:
        try:
            X_train, X_test, y_train, y_test = load_dataset(dataset)
            evaluate_models(dataset, X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f"Failed to process {dataset}: {str(e)}")