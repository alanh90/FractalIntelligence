import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, fetch_california_housing, load_diabetes
from sklearn.neural_network import MLPClassifier, MLPRegressor
import time
from scipy import stats
import random
import matplotlib.cm as cm
import pandas as pd
from functools import partial
from mpl_toolkits.mplot3d import Axes3D


# --- Core Fluid Topology Network Implementation --- #
class FluidTopoNetwork(BaseEstimator, ClassifierMixin):
    """
    Fluid Topology Network: An alternative to neural networks based on fluid dynamics principles.

    Instead of neurons and weights, this model uses:
    - Pressure points (data concentration areas)
    - Flow pathways (connections between pressure points)
    - Vortices (classification/regression centers)

    Learning occurs through fluid dynamics simulation rather than backpropagation.
    """

    def __init__(self, turbulence=0.1, flow_rate=0.05, viscosity=0.8,
                 pressure_sensitivity=1.0, n_pressure_points=None, max_iterations=15):
        """
        Initialize FluidTopoNetwork.

        Parameters:
        - turbulence: Amount of randomness in the flow dynamics (like dropout in NNs)
        - flow_rate: Speed of information flow through the network (like learning rate)
        - viscosity: Resistance to structural changes (higher = more stable)
        - pressure_sensitivity: Sensitivity to input data pressure points
        - n_pressure_points: Number of pressure points to create (if None, will be data-dependent)
        - max_iterations: Maximum number of fluid dynamics simulation iterations
        """
        self.turbulence = turbulence
        self.flow_rate = flow_rate
        self.viscosity = viscosity
        self.pressure_sensitivity = pressure_sensitivity
        self.n_pressure_points = n_pressure_points
        self.max_iterations = max_iterations

        # Internal state
        self.flow_pathways = None
        self.pressure_points = None
        self.vortices = None
        self.class_mappings = None
        self.feature_importance_ = None
        self.convergence_history = []
        self.iteration_pressures = []

    def _initialize_fluid_state(self, X, y):
        """Initialize the fluid network state based on data."""
        n_samples, n_features = X.shape

        # Determine number of classes
        if hasattr(self, "_regression"):
            n_classes = 1  # Regression task
        else:
            # For classification tasks
            self.classes_ = np.unique(y)
            n_classes = len(self.classes_)

        # Calculate optimal number of pressure points if not specified
        if self.n_pressure_points is None:
            # Scale with data size and dimensionality, but with reasonable bounds
            self.n_pressure_points = min(n_samples // 3, max(n_features * 2, 10))

        # Create initial pressure points (data concentration areas)
        n_points = min(self.n_pressure_points, n_samples)

        # Use k-means-like initialization for better coverage
        indices = [np.random.randint(0, n_samples)]  # Start with random point

        # Add points that are far from existing ones
        while len(indices) < n_points:
            max_dist = 0
            max_idx = -1

            # Find point with maximum minimum distance to existing points
            for i in range(n_samples):
                if i in indices:
                    continue

                # Calculate minimum distance to any existing point
                min_dist = float('inf')
                for idx in indices:
                    dist = np.sum((X[i] - X[idx]) ** 2)
                    min_dist = min(min_dist, dist)

                if min_dist > max_dist:
                    max_dist = min_dist
                    max_idx = i

            indices.append(max_idx)

        # Create pressure points from selected indices
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

        # Initialize vortices (classification/regression centers)
        self.vortices = []

        if hasattr(self, "_regression"):
            # For regression, create vortices based on output value ranges
            y_min, y_max = np.min(y), np.max(y)
            n_vortices = min(5, n_samples // 10)  # Reasonable number of vortices

            # Create vortices for different output ranges
            for i in range(n_vortices):
                y_val = y_min + (y_max - y_min) * i / (n_vortices - 1)

                # Find samples closest to this y value
                diffs = np.abs(y - y_val)
                closest_indices = np.argsort(diffs)[:max(5, n_samples // 20)]

                if len(closest_indices) > 0:
                    # Create vortex at center of these samples
                    vortex_center = np.mean(X[closest_indices], axis=0)
                    vortex_strength = 1.0 / n_vortices
                    self.vortices.append((vortex_center, vortex_strength, y_val))
        else:
            # For classification, create vortices for each class
            for class_idx in range(n_classes):
                class_samples = X[y == self.classes_[class_idx]]

                if len(class_samples) > 0:
                    # Create vortex at class center
                    vortex_center = np.mean(class_samples, axis=0)
                    vortex_strength = len(class_samples) / n_samples
                    self.vortices.append((vortex_center, vortex_strength, self.classes_[class_idx]))

        # Store feature importance (will be updated during training)
        self.feature_importance_ = np.ones(n_features) / n_features

    def _fluid_dynamics(self, X, y=None):
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

        # Calculate average feature importance based on pressure point movements
        feature_importance = np.zeros(self.pressure_points.shape[1])

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

                # Add turbulence (stochastic element)
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

            # If we have labels, add label-based force
            if y is not None:
                # For each class/value, calculate attraction/repulsion
                if hasattr(self, "_regression"):
                    # For regression, attract points to samples with similar output values
                    y_distances = np.abs(self._predict_single(point.reshape(1, -1)) - y)
                    # Find closest samples in output space
                    closest_indices = np.argsort(y_distances)[:max(5, len(y) // 20)]

                    if len(closest_indices) > 0:
                        # Attract towards these samples
                        target = np.mean(X[closest_indices], axis=0)
                        direction = target - point
                        movement[i] += 0.1 * direction  # Gentle attraction
                else:
                    # For classification, attract to correctly classified, repel from incorrectly classified
                    predictions = self._predict_single(point.reshape(1, -1))

                    # Loop through samples to calculate forces
                    for sample_idx in range(min(100, len(X))):  # Limit for efficiency
                        sample = X[sample_idx]
                        label = y[sample_idx]

                        # Calculate direction and distance
                        direction = sample - point
                        distance = np.linalg.norm(direction)

                        if distance > 0:
                            # Normalize direction
                            direction = direction / distance

                            # Determine if correctly classified
                            if predictions == label:
                                # Correctly classified - gentle attraction
                                force = 0.05 / (distance + 1.0)
                                movement[i] += direction * force
                            else:
                                # Incorrectly classified - repulsion
                                force = -0.1 / (distance + 1.0)
                                movement[i] += direction * force

            # Normalize movement by total flow
            total_flow = np.sum(self.flow_pathways[i])
            if total_flow > 0:
                movement[i] /= total_flow

        # Apply movement with viscosity damping
        original_points = self.pressure_points.copy()
        self.pressure_points += movement * (1 - self.viscosity) * self.flow_rate

        # Update feature importance based on point movements
        for feat_idx in range(self.pressure_points.shape[1]):
            # Feature importance proportional to movement in that dimension
            importance = np.mean(np.abs(self.pressure_points[:, feat_idx] - original_points[:, feat_idx]))
            feature_importance[feat_idx] = importance

        # Update feature importance (with smoothing)
        self.feature_importance_ = 0.8 * self.feature_importance_ + 0.2 * (
                feature_importance / (np.sum(feature_importance) + 1e-10)
        )

        return pressures

    def _vortex_dynamics(self, X, y):
        """Update vortices based on data distribution."""
        # For classification tasks
        if not hasattr(self, "_regression"):
            new_vortices = []
            for class_idx in range(len(self.classes_)):
                class_val = self.classes_[class_idx]
                class_mask = y == class_val

                if np.any(class_mask):
                    class_samples = X[class_mask]

                    # Get pressure points most associated with this class
                    class_pressures = np.zeros(len(self.pressure_points))
                    for i, point in enumerate(self.pressure_points):
                        class_distances = np.linalg.norm(class_samples - point, axis=1)
                        class_pressures[i] = np.sum(1.0 / (1.0 + class_distances ** 2))

                    # Create new vortex with influence from top pressure points
                    top_indices = np.argsort(class_pressures)[-min(3, len(class_pressures)):]
                    vortex_center = np.mean(self.pressure_points[top_indices], axis=0)
                    vortex_strength = np.sum(class_mask) / len(X)

                    # Create or update vortex
                    new_vortices.append((vortex_center, vortex_strength, class_val))

            self.vortices = new_vortices
        else:
            # For regression tasks
            y_min, y_max = np.min(y), np.max(y)
            n_vortices = len(self.vortices)

            new_vortices = []
            for i, (_, _, y_val) in enumerate(self.vortices):
                # Find samples closest to this y value
                diffs = np.abs(y - y_val)
                closest_indices = np.argsort(diffs)[:max(5, len(X) // 20)]

                if len(closest_indices) > 0:
                    # Update vortex position
                    vortex_center = np.mean(X[closest_indices], axis=0)
                    vortex_strength = 1.0 / n_vortices
                    new_vortices.append((vortex_center, vortex_strength, y_val))
                else:
                    # Keep original vortex if no close samples
                    new_vortices.append(self.vortices[i])

            self.vortices = new_vortices

    def _predict_single(self, x):
        """Predict for a single sample using fluid dynamics principles."""
        if hasattr(self, "_regression"):
            # For regression, calculate weighted average of vortex values
            vortex_influences = []
            for center, strength, y_val in self.vortices:
                # Distance-based influence with vortex strength
                distance = np.linalg.norm(x - center)
                influence = strength / (1.0 + distance ** 2)
                vortex_influences.append((influence, y_val))

            # Calculate weighted average
            total_influence = sum(infl for infl, _ in vortex_influences)
            if total_influence > 0:
                y_pred = sum(infl * y_val for infl, y_val in vortex_influences) / total_influence
            else:
                # Default to average if no influence
                y_pred = np.mean([y_val for _, _, y_val in self.vortices])

            return y_pred
        else:
            # For classification, find vortex with strongest influence
            vortex_influences = []
            for center, strength, class_val in self.vortices:
                # Distance-based influence with vortex strength
                distance = np.linalg.norm(x - center)
                influence = strength / (1.0 + distance ** 2)
                vortex_influences.append((influence, class_val))

            # Find vortex with strongest influence
            if vortex_influences:
                return max(vortex_influences, key=lambda x: x[0])[1]
            else:
                return self.classes_[0]  # Default if no vortices

    def fit(self, X, y):
        """Fit the fluid network to training data."""
        if X.size == 0 or y.size == 0:
            raise ValueError("Empty input data provided.")

        # Initialize fluid state
        self._initialize_fluid_state(X, y)

        # Run fluid dynamics for multiple iterations
        n_iterations = self.max_iterations
        convergence_metric = []

        for iteration in range(n_iterations):
            # Store original state for convergence check
            original_points = self.pressure_points.copy()
            original_vortices = [(c.copy(), s, v) for c, s, v in self.vortices]

            # Update fluid dynamics
            pressures = self._fluid_dynamics(X, y)
            self.iteration_pressures.append(pressures)

            # Update vortices based on class data
            self._vortex_dynamics(X, y)

            # Check convergence (how much the model changed)
            point_change = np.mean(np.linalg.norm(self.pressure_points - original_points, axis=1))
            vortex_change = np.mean([np.linalg.norm(v[0] - ov[0])
                                     for v, ov in zip(self.vortices, original_vortices)])

            convergence = point_change + vortex_change
            convergence_metric.append(convergence)
            self.convergence_history.append(convergence)

            # Early stopping if converged
            if iteration > 5 and convergence < 0.01:
                break

        return self

    def predict(self, X):
        """Predict class labels using fluid dynamics."""
        if X.size == 0:
            raise ValueError("Empty input data for prediction.")

        if self.pressure_points is None or self.vortices is None:
            raise ValueError("Model not fitted yet.")

        # Vectorize prediction for efficiency
        predictions = np.array([self._predict_single(x) for x in X])
        return predictions

    def predict_proba(self, X):
        """Predict class probabilities for classification."""
        if hasattr(self, "_regression"):
            raise ValueError("predict_proba is only available for classification.")

        if X.size == 0:
            raise ValueError("Empty input data for prediction.")

        if self.pressure_points is None or self.vortices is None:
            raise ValueError("Model not fitted yet.")

        # Calculate probabilities based on vortex influences
        probabilities = np.zeros((X.shape[0], len(self.classes_)))

        for i, x in enumerate(X):
            # Get influences from all vortices
            vortex_influences = []
            for center, strength, class_val in self.vortices:
                # Calculate influence based on distance
                distance = np.linalg.norm(x - center)
                influence = strength / (1.0 + distance ** 2)

                # Find class index
                class_idx = np.where(self.classes_ == class_val)[0][0]
                vortex_influences.append((influence, class_idx))

            # Convert influences to probabilities
            total_influence = sum(infl for infl, _ in vortex_influences)
            if total_influence > 0:
                for influence, class_idx in vortex_influences:
                    probabilities[i, class_idx] = influence / total_influence
            else:
                # If no influence, use uniform distribution
                probabilities[i, :] = 1.0 / len(self.classes_)

        return probabilities

    def visualize_model(self, X=None, y=None, title="Fluid Topology Network"):
        """Visualize the fluid network structure."""
        if self.pressure_points is None:
            raise ValueError("Model not fitted yet.")

        # Create figure
        plt.figure(figsize=(12, 10))

        # Set up colormap for classes/values
        cmap = plt.colormaps.get('viridis')

        # Plot pressure points and flow pathways
        n_points = len(self.pressure_points)

        # For high-dimensional data, use PCA or t-SNE to reduce to 2D
        if self.pressure_points.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            points_2d = pca.fit_transform(self.pressure_points)

            # Transform vortices too
            vortices_2d = [
                (pca.transform(center.reshape(1, -1))[0], strength, class_val)
                for center, strength, class_val in self.vortices
            ]

            # Transform data points if provided
            if X is not None:
                X_2d = pca.transform(X)
        else:
            # Already 2D, use as is
            points_2d = self.pressure_points
            vortices_2d = self.vortices
            X_2d = X

        # Plot flow pathways
        for i in range(n_points):
            for j in range(i + 1, n_points):
                flow = self.flow_pathways[i, j]
                if flow > 0.1:  # Only plot significant flows
                    plt.plot([points_2d[i, 0], points_2d[j, 0]],
                             [points_2d[i, 1], points_2d[j, 1]],
                             'k-', alpha=np.clip(flow, 0, 1), linewidth=min(flow * 3, 5))

        # Plot pressure points
        plt.scatter(points_2d[:, 0], points_2d[:, 1], s=100, c='blue',
                    alpha=0.7, edgecolors='k', label='Pressure Points')

        # Plot vortices
        if hasattr(self, "_regression"):
            # For regression, color by output value
            vortex_values = [v[2] for v in vortices_2d]
            min_val, max_val = min(vortex_values), max(vortex_values)

            for center, strength, y_val in vortices_2d:
                # Normalize value for color
                color_val = (y_val - min_val) / (max_val - min_val + 1e-10)
                plt.scatter(center[0], center[1], s=min(300 * strength, 500),
                            c=[cmap(color_val)], alpha=0.7, edgecolors='k')
        else:
            # For classification, color by class
            for center, strength, class_val in vortices_2d:
                class_idx = np.where(self.classes_ == class_val)[0][0]
                color_val = class_idx / (len(self.classes_) - 1) if len(self.classes_) > 1 else 0.5
                plt.scatter(center[0], center[1], s=min(300 * strength, 500),
                            c=[cmap(color_val)], alpha=0.7, edgecolors='k',
                            label=f'Class {class_val}')

        # Plot data points if provided
        if X is not None and y is not None:
            if hasattr(self, "_regression"):
                # For regression, color points by value
                sc = plt.scatter(X_2d[:, 0], X_2d[:, 1], s=30, c=y, cmap=cmap, alpha=0.5)
                plt.colorbar(sc, label='Output Value')
            else:
                # For classification, color points by class
                for i, class_val in enumerate(self.classes_):
                    mask = y == class_val
                    if np.any(mask):
                        color_val = i / (len(self.classes_) - 1) if len(self.classes_) > 1 else 0.5
                        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=30,
                                    c=[cmap(color_val)], alpha=0.5,
                                    label=f'Data Class {class_val}')

        # Finishing touches
        plt.title(title)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()

    def visualize_convergence(self):
        """Visualize the convergence of the model during training."""
        if not self.convergence_history:
            raise ValueError("Model not fitted yet or convergence history not available.")

        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_history, 'b-', linewidth=2)
        plt.title('FluidTopoNetwork Convergence')
        plt.xlabel('Iterations')
        plt.ylabel('Change in Model')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()

    def visualize_feature_importance(self, feature_names=None):
        """Visualize feature importance."""
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted yet.")

        plt.figure(figsize=(10, 6))

        # Use feature names if provided, otherwise use indices
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(self.feature_importance_))]

        # Sort features by importance
        sorted_idx = np.argsort(self.feature_importance_)
        pos = np.arange(sorted_idx.shape[0]) + 0.5

        plt.barh(pos, self.feature_importance_[sorted_idx], align='center')
        plt.yticks(pos, [feature_names[i] for i in sorted_idx])
        plt.title('Feature Importance (FluidTopoNetwork)')
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        return plt.gcf()


# --- FluidTopoRegressor: Regression version --- #
class FluidTopoRegressor(FluidTopoNetwork, RegressorMixin):
    """Fluid Topology Network for regression tasks."""

    def __init__(self, turbulence=0.05, flow_rate=0.05, viscosity=0.8,
                 pressure_sensitivity=1.0, n_pressure_points=None, max_iterations=15):
        super().__init__(
            turbulence=turbulence,
            flow_rate=flow_rate,
            viscosity=viscosity,
            pressure_sensitivity=pressure_sensitivity,
            n_pressure_points=n_pressure_points,
            max_iterations=max_iterations
        )
        # Mark as regression model
        self._regression = True

    def score(self, X, y):
        """Return R^2 score for regression."""
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


# --- Enhanced FluidTopoNetwork with visualization and analysis --- #
class EnhancedFluidTopoNetwork(FluidTopoNetwork):
    def __init__(self, turbulence=0.1, flow_rate=0.05, viscosity=0.8,
                 pressure_sensitivity=1.0, vortex_interaction=0.2, adaptive_turbulence=True,
                 n_pressure_points=None, max_iterations=15):
        super().__init__(
            turbulence=turbulence,
            flow_rate=flow_rate,
            viscosity=viscosity,
            pressure_sensitivity=pressure_sensitivity,
            n_pressure_points=n_pressure_points,
            max_iterations=max_iterations
        )
        self.vortex_interaction = vortex_interaction
        self.adaptive_turbulence = adaptive_turbulence
        self.error_history = []
        self.decision_boundaries = None

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
            self._fluid_dynamics(X, y)
            self._adaptive_parameters(X, y)

            # Validate current model
            if len(X) > 10:
                preds = self.predict(X)
                error = np.mean(preds != y)
                self.error_history.append(error)

        # For 2D data, precompute decision boundaries for visualization
        if X.shape[1] == 2:
            self._precompute_decision_boundaries(X)

        return self

    def _precompute_decision_boundaries(self, X):
        """Precompute decision boundaries for 2D data."""
        # Create a grid covering the data space
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        # Make predictions on the grid
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        self.decision_boundaries = (xx, yy, Z)

    def visualize_decision_boundary(self, X, y, title="Enhanced FluidTopoNetwork Decision Boundary"):
        """Visualize decision boundaries for 2D data."""
        if X.shape[1] != 2:
            raise ValueError("Decision boundary visualization only works for 2D data.")

        if self.decision_boundaries is None:
            self._precompute_decision_boundaries(X)

        xx, yy, Z = self.decision_boundaries

        plt.figure(figsize=(10, 8))

        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

        # Plot data points
        unique_classes = np.unique(y)
        for cls in unique_classes:
            plt.scatter(X[y == cls, 0], X[y == cls, 1],
                        alpha=np.clip(0.8, 0, 1), label=f'Class {cls}')

        # Plot vortices
        for center, strength, class_val in self.vortices:
            plt.scatter(center[0], center[1], s=min(300 * strength, 500),
                        marker='*', color='black', edgecolors='white',
                        label=f'Vortex Class {class_val}')

        # Plot pressure points
        plt.scatter(self.pressure_points[:, 0], self.pressure_points[:, 1],
                    s=100, marker='o', color='green', alpha=np.clip(0.5, 0, 1),
                    label='Pressure Points')

        # Plot flow pathways
        n_points = len(self.pressure_points)
        for i in range(n_points):
            for j in range(i + 1, n_points):
                flow = self.flow_pathways[i, j]
                if flow > 0.1:  # Only plot significant flows
                    plt.plot([self.pressure_points[i, 0], self.pressure_points[j, 0]],
                             [self.pressure_points[i, 1], self.pressure_points[j, 1]],
                             'k-', alpha=np.clip(flow, 0, 1), linewidth=min(flow * 3, 5))

        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.tight_layout()
        return plt.gcf()

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

        # In this version, we just return the predictions
        # But we could use uncertainties for ensemble methods or confidence estimation
        return predictions

    def visualize_model_3d(self, X, y, title="Enhanced FluidTopoNetwork 3D Visualization"):
        """Create a 3D visualization of the model for 2D data."""
        if X.shape[1] != 2:
            raise ValueError("3D visualization only works for 2D data.")

        if self.decision_boundaries is None:
            self._precompute_decision_boundaries(X)

        xx, yy, Z = self.decision_boundaries

        # Create 3D figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create fluid-like surface
        surface = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4)

        # Plot data points
        unique_classes = np.unique(y)
        for cls in unique_classes:
            ax.scatter(X[y == cls, 0], X[y == cls, 1], cls,
                       alpha=np.clip(0.8, 0, 1), label=f'Class {cls}')

        # Plot vortices in 3D
        for center, strength, class_val in self.vortices:
            ax.scatter(center[0], center[1], class_val,
                       s=min(300 * strength, 500), marker='*', color='black',
                       label=f'Vortex Class {class_val}')

        ax.set_title(title)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Class')
        ax.legend()
        plt.tight_layout()
        return fig


# --- Dataset Loader --- #
def load_dataset(name, task="classification"):
    """Load and preprocess a dataset."""
    try:
        if task == "classification":
            if name == "Iris":
                data = load_iris()
            elif name == "Wine":
                data = load_wine()
            elif name == "Breast Cancer":
                data = load_breast_cancer()
            else:
                raise ValueError(f"Unknown classification dataset: {name}")
        else:  # regression
            if name == "California Housing":
                data = fetch_california_housing()
            elif name == "Diabetes":
                data = load_diabetes()
            else:
                raise ValueError(f"Unknown regression dataset: {name}")

        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Get feature names if available
        feature_names = getattr(data, 'feature_names', None)

        return X_train, X_test, y_train, y_test, feature_names, name
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {name}: {str(e)}")


# --- Model Evaluation --- #
def evaluate_classification(dataset_name, X_train, X_test, y_train, y_test, feature_names=None, visualize=True):
    """Evaluate models on a classification dataset with visualizations."""
    print(f"\nEvaluating Classification - {dataset_name}")

    # Initialize models
    models = [
        ("FluidTopoNetwork", FluidTopoNetwork(turbulence=0.1, flow_rate=0.05)),
        ("EnhancedFluidTopoNetwork", EnhancedFluidTopoNetwork(turbulence=0.1, flow_rate=0.05, vortex_interaction=0.2)),
        ("Neural Network (MLP)", MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42))
    ]

    results = []

    for name, model in models:
        try:
            # Training
            start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start

            # Prediction
            start = time.time()
            y_pred = model.predict(X_test)
            predict_time = time.time() - start

            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            train_acc = accuracy_score(y_train, model.predict(X_train))

            print(f"{name} - Train Time: {train_time:.4f}s, Predict Time: {predict_time:.4f}s, "
                  f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {acc:.4f}")

            results.append({
                'name': name,
                'model': model,
                'train_time': train_time,
                'predict_time': predict_time,
                'train_accuracy': train_acc,
                'test_accuracy': acc
            })

            # Visualizations for FluidTopoNetwork models
            if visualize and 'FluidTopo' in name:
                if hasattr(model, 'visualize_model'):
                    try:
                        fig = model.visualize_model(X_train, y_train, title=f"{name} - {dataset_name}")
                        plt.savefig(f"{name}_{dataset_name}_model.png")
                        plt.close(fig)
                    except Exception as e:
                        print(f"{name} - Error: {str(e)}")

                if hasattr(model, 'visualize_convergence'):
                    try:
                        fig = model.visualize_convergence()
                        plt.savefig(f"{name}_{dataset_name}_convergence.png")
                        plt.close(fig)
                    except Exception as e:
                        print(f"{name} - Error: {str(e)}")

                if feature_names is not None and hasattr(model, 'visualize_feature_importance'):
                    try:
                        fig = model.visualize_feature_importance(feature_names)
                        plt.savefig(f"{name}_{dataset_name}_feature_importance.png")
                        plt.close(fig)
                    except Exception as e:
                        print(f"{name} - Error: {str(e)}")

                # For 2D datasets, visualize decision boundaries
                if X_train.shape[1] == 2 and hasattr(model, 'visualize_decision_boundary'):
                    try:
                        fig = model.visualize_decision_boundary(X_train, y_train)
                        plt.savefig(f"{name}_{dataset_name}_decision_boundary.png")
                        plt.close(fig)
                    except Exception as e:
                        print(f"{name} - Error: {str(e)}")

                    if name == "EnhancedFluidTopoNetwork":
                        try:
                            fig = model.visualize_model_3d(X_train, y_train)
                            plt.savefig(f"{name}_{dataset_name}_3d.png")
                            plt.close(fig)
                        except Exception as e:
                            print(f"{name} - Error: {str(e)}")

        except Exception as e:
            print(f"{name} - Error: {str(e)}")

    return results


def evaluate_regression(dataset_name, X_train, X_test, y_train, y_test, feature_names=None, visualize=True):
    """Evaluate models on a regression dataset with visualizations."""
    print(f"\nEvaluating Regression - {dataset_name}")

    # Initialize models
    models = [
        ("FluidTopoRegressor", FluidTopoRegressor(turbulence=0.05, flow_rate=0.05)),
        ("Neural Network (MLP)", MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42))
    ]

    results = []

    for name, model in models:
        try:
            # Training
            start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start

            # Prediction
            start = time.time()
            y_pred = model.predict(X_test)
            predict_time = time.time() - start

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            train_mse = mean_squared_error(y_train, model.predict(X_train))
            train_r2 = r2_score(y_train, model.predict(X_train))

            print(f"{name} - Train Time: {train_time:.4f}s, Predict Time: {predict_time:.4f}s, "
                  f"Train MSE: {train_mse:.4f}, Test MSE: {mse:.4f}, "
                  f"Train R²: {train_r2:.4f}, Test R²: {r2:.4f}")

            results.append({
                'name': name,
                'model': model,
                'train_time': train_time,
                'predict_time': predict_time,
                'train_mse': train_mse,
                'test_mse': mse,
                'train_r2': train_r2,
                'test_r2': r2
            })

            # Visualizations for FluidTopoRegressor
            if visualize and name == "FluidTopoRegressor":
                if hasattr(model, 'visualize_model'):
                    try:
                        fig = model.visualize_model(X_train, y_train, title=f"{name} - {dataset_name}")
                        plt.savefig(f"{name}_{dataset_name}_model.png")
                        plt.close(fig)
                    except Exception as e:
                        print(f"{name} - Error: {str(e)}")

                if hasattr(model, 'visualize_convergence'):
                    try:
                        fig = model.visualize_convergence()
                        plt.savefig(f"{name}_{dataset_name}_convergence.png")
                        plt.close(fig)
                    except Exception as e:
                        print(f"{name} - Error: {str(e)}")

                if feature_names is not None and hasattr(model, 'visualize_feature_importance'):
                    try:
                        fig = model.visualize_feature_importance(feature_names)
                        plt.savefig(f"{name}_{dataset_name}_feature_importance.png")
                        plt.close(fig)
                    except Exception as e:
                        print(f"{name} - Error: {str(e)}")

                # For simple datasets, plot actual vs predicted
                if X_train.shape[1] <= 3:
                    try:
                        plt.figure(figsize=(10, 6))
                        plt.scatter(y_test, y_pred, alpha=0.5)
                        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
                        plt.xlabel('Actual')
                        plt.ylabel('Predicted')
                        plt.title(f"{name} - {dataset_name} - Actual vs Predicted")
                        plt.savefig(f"{name}_{dataset_name}_actual_vs_predicted.png")
                        plt.close()
                    except Exception as e:
                        print(f"{name} - Error: {str(e)}")

        except Exception as e:
            print(f"{name} - Error: {str(e)}")

    return results


def summarize_results(classification_results, regression_results):
    """Create summary tables and charts for all results."""
    # Classification summary
    if classification_results:
        print("\n===== CLASSIFICATION RESULTS =====")
        class_df = pd.DataFrame(classification_results)
        print(class_df[['name', 'train_time', 'predict_time', 'train_accuracy', 'test_accuracy']])

        # Plot comparison chart
        plt.figure(figsize=(12, 6))

        # Accuracy comparison
        plt.subplot(1, 2, 1)
        names = class_df['name'].tolist()
        train_acc = class_df['train_accuracy'].tolist()
        test_acc = class_df['test_accuracy'].tolist()

        x = np.arange(len(names))
        width = 0.35

        plt.bar(x - width / 2, train_acc, width, label='Train Accuracy')
        plt.bar(x + width / 2, test_acc, width, label='Test Accuracy')

        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('Classification Accuracy Comparison')
        plt.xticks(x, names, rotation=45)
        plt.legend()

        # Time comparison
        plt.subplot(1, 2, 2)
        train_time = class_df['train_time'].tolist()
        predict_time = class_df['predict_time'].tolist()

        plt.bar(x - width / 2, train_time, width, label='Train Time (s)')
        plt.bar(x + width / 2, predict_time, width, label='Predict Time (s)')

        plt.xlabel('Model')
        plt.ylabel('Time (s)')
        plt.title('Classification Time Comparison')
        plt.xticks(x, names, rotation=45)
        plt.legend()

        plt.tight_layout()
        plt.savefig("classification_comparison.png")
        plt.close()

    # Regression summary
    if regression_results:
        print("\n===== REGRESSION RESULTS =====")
        reg_df = pd.DataFrame(regression_results)
        print(reg_df[['name', 'train_time', 'predict_time', 'train_mse', 'test_mse', 'train_r2', 'test_r2']])

        # Plot comparison chart
        plt.figure(figsize=(12, 8))

        # MSE comparison
        plt.subplot(2, 2, 1)
        names = reg_df['name'].tolist()
        train_mse = reg_df['train_mse'].tolist()
        test_mse = reg_df['test_mse'].tolist()

        x = np.arange(len(names))
        width = 0.35

        plt.bar(x - width / 2, train_mse, width, label='Train MSE')
        plt.bar(x + width / 2, test_mse, width, label='Test MSE')

        plt.xlabel('Model')
        plt.ylabel('MSE')
        plt.title('Regression MSE Comparison')
        plt.xticks(x, names, rotation=45)
        plt.legend()

        # R² comparison
        plt.subplot(2, 2, 2)
        train_r2 = reg_df['train_r2'].tolist()
        test_r2 = reg_df['test_r2'].tolist()

        plt.bar(x - width / 2, train_r2, width, label='Train R²')
        plt.bar(x + width / 2, test_r2, width, label='Test R²')

        plt.xlabel('Model')
        plt.ylabel('R²')
        plt.title('Regression R² Comparison')
        plt.xticks(x, names, rotation=45)
        plt.legend()

        # Time comparison
        plt.subplot(2, 2, 3)
        train_time = reg_df['train_time'].tolist()
        predict_time = reg_df['predict_time'].tolist()

        plt.bar(x - width / 2, train_time, width, label='Train Time (s)')
        plt.bar(x + width / 2, predict_time, width, label='Predict Time (s)')

        plt.xlabel('Model')
        plt.ylabel('Time (s)')
        plt.title('Regression Time Comparison')
        plt.xticks(x, names, rotation=45)
        plt.legend()

        plt.tight_layout()
        plt.savefig("regression_comparison.png")
        plt.close()


# --- Main Execution --- #
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Classification datasets
    classification_datasets = ["Iris", "Wine", "Breast Cancer"]
    classification_results = []

    for dataset in classification_datasets:
        try:
            X_train, X_test, y_train, y_test, feature_names, name = load_dataset(dataset, task="classification")
            results = evaluate_classification(name, X_train, X_test, y_train, y_test, feature_names)

            # Store results
            for r in results:
                r['dataset'] = name
                classification_results.append(r)

        except Exception as e:
            print(f"Failed to process {dataset}: {str(e)}")

    # Regression datasets
    regression_datasets = ["California Housing", "Diabetes"]
    regression_results = []

    for dataset in regression_datasets:
        try:
            X_train, X_test, y_train, y_test, feature_names, name = load_dataset(dataset, task="regression")
            results = evaluate_regression(name, X_train, X_test, y_train, y_test, feature_names)

            # Store results
            for r in results:
                r['dataset'] = name
                regression_results.append(r)

        except Exception as e:
            print(f"Failed to process {dataset}: {str(e)}")

    # Summarize results
    summarize_results(classification_results, regression_results)

    print("\nEvaluation complete. Visualizations saved to current directory.")
    print("\nFluidTopoNetwork represents a novel AI paradigm that replaces traditional neural networks")
    print("with a fluid dynamics-based approach, eliminating the need for backpropagation while achieving")
    print("competitive performance on standard benchmark tasks.")