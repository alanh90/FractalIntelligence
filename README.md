# MandelaNet

## Overview

**MandelaNet** is an innovative, fractal-inspired adaptive learning algorithm designed as an alternative to traditional neural networks. Drawing inspiration from concepts such as mandala fractals, flow dynamics, and adaptive “box” representations, this approach starts with a single abstract "box" that encodes the entire data space and recursively subdivides it into radially symmetric child nodes. Each node adapts locally using simple momentum-based updates and “liquid splitting” when error thresholds are exceeded—all without relying on global backpropagation.

### Key Features
- **Adaptive Structure:** Begins with a single seed node ("box") that evolves into a complex, fractal-like network based on data complexity.
- **Fractal Adjusters:** Uses local momentum-based adjustments to update node centers, enabling rapid and dynamic adaptation.
- **Flow Dynamics:** Implements softmax-like weighting to guide data flow through the network, mimicking how water flows in a river or patterns in a mandala.
- **Non-Backpropagation Learning:** Relies on local error feedback and adaptive splits instead of global gradient descent.

## Experimental Results

The algorithm was compared with several fast-learning models on standard datasets (Iris, Wine, and Breast Cancer):

### Iris Dataset
- **MandelaNet:** 0.0040 s, **100% Accuracy**
- **Decision Tree:** 0.0015 s, 100% Accuracy
- **Random Forest:** 0.0496 s, 100% Accuracy
- **Logistic Regression:** 0.0010 s, 100% Accuracy
- **SVC:** 0.0000 s, 100% Accuracy

### Wine Dataset
- **MandelaNet:** 0.0045 s, **100% Accuracy**
- **Decision Tree:** 0.0000 s, 94.44% Accuracy
- **Random Forest:** 0.0564 s, 100% Accuracy
- **Logistic Regression:** 0.0010 s, 100% Accuracy
- **SVC:** 0.0000 s, 100% Accuracy

### Breast Cancer Dataset
- **MandelaNet:** 0.0211 s, **100% Accuracy**
- **Decision Tree:** 0.0045 s, 94.74% Accuracy
- **Random Forest:** 0.1023 s, 96.49% Accuracy
- **Logistic Regression:** 0.0064 s, 97.37% Accuracy
- **SVC:** 0.0010 s, 98.25% Accuracy

*Note: Results on these small, well-known benchmarks are promising, particularly showing fast training times and high accuracy on Iris and Wine. On Breast Cancer, MandelaNet achieved 100% accuracy in this run, which is notably competitive with standard methods.*
