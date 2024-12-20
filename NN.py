'''
================================================================================
SUMMARY OF NN RESULTS
================================================================================
      Dataset  NN Balanced Accuracy  NN Training Time (s)  NN Prediction Time (s)
        Moons              0.990000                0.4313                   0.001
      Circles              0.890000                0.2153                   0.001
       Digits              0.974683                0.4992                   0.001
         Iris              0.966667                0.0846                   0.000
Breast Cancer              0.967262                0.1998                   0.000
         Wine              0.966667                0.0621                   0.000
'''

import numpy as np
import time
from sklearn.datasets import make_moons, make_circles, load_digits, load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier  # Multi-layer Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
import pandas as pd

def test_on_dataset_nn(name: str, X: np.ndarray, y: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    start_time = time.time()
    nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=700, random_state=42) # Adjust hidden layers and max_iter
    nn.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"NN Training time: {training_time:.4f} seconds")

    start_time = time.time()
    predictions = nn.predict(X_test)
    prediction_time = time.time() - start_time
    print(f"NN Prediction time: {prediction_time:.4f} seconds")

    balanced_acc = balanced_accuracy_score(y_test, predictions)
    print(f"NN Balanced Accuracy: {balanced_acc:.4f}")
    return balanced_acc, training_time, prediction_time

def run_nn_experiments():
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
        print(f"\nTesting NN on {dataset_name} dataset...")
        X, y = dataset_loader()
        balanced_acc, training_time, prediction_time = test_on_dataset_nn(dataset_name, X, y)
        results.append(
                {
                    "Dataset": dataset_name,
                    "NN Balanced Accuracy": balanced_acc,
                    "NN Training Time (s)": round(training_time, 4),
                    "NN Prediction Time (s)": round(prediction_time, 4),
                }
            )
    print("\n" + "=" * 80)
    print("SUMMARY OF NN RESULTS")
    print("=" * 80)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    run_nn_experiments()