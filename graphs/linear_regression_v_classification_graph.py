import numpy as np
import matplotlib.pyplot as plt

# Generate data for linear regression
x_reg = np.array([0, 1, 2, 3, 4, 5])
y_reg = np.array([1.2, 2.1, 2.9, 3.8, 5.1, 5.9])  # continuous targets

# Fit a simple linear regression line y = w*x + b
w_reg = np.polyfit(x_reg, y_reg, 1)
y_pred_reg = np.polyval(w_reg, x_reg)

# Generate data for classification
x_clf = np.array([0, 1, 2, 3, 4, 5])
y_clf = np.array([0, 0, 0, 1, 1, 1])  # class labels 0 or 1

# Logistic function for decision boundary: w*x + b = 0
# We find weights roughly separating the classes for illustration
w_clf = 1.0
b_clf = -2.5

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Linear regression plot
axs[0].scatter(x_reg, y_reg, color='blue', label='Data points')
axs[0].plot(x_reg, y_pred_reg, color='red', label='Regression line')
axs[0].set_title('Linear Regression (MSE)')
axs[0].set_xlabel('Input x')
axs[0].set_ylabel('Output y')
axs[0].legend()
axs[0].grid(True)

# Classification plot
# Points colored by class
axs[1].scatter(x_clf[y_clf == 0], y_clf[y_clf == 0], color='blue', label='Class 0')
axs[1].scatter(x_clf[y_clf == 1], y_clf[y_clf == 1], color='red', label='Class 1')

# Decision boundary at sigmoid(w*x + b) = 0.5 => w*x + b = 0
decision_boundary = -b_clf / w_clf
axs[1].axvline(x=decision_boundary, color='green', linestyle='--', label='Decision boundary')

axs[1].set_ylim(-0.2, 1.2)
axs[1].set_title('Binary Classification (Cross-Entropy)')
axs[1].set_xlabel('Input x')
axs[1].set_ylabel('Class label')
axs[1].legend()
axs[1].grid(True)

plt.show()
