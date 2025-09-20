import numpy as np
import matplotlib.pyplot as plt

# Weight values from -10 to 10
w = np.linspace(-10, 10, 400)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# True label
y = 1

# Prediction
y_hat = sigmoid(w)  # since x=1 and bias=0

# Cross-entropy loss (for y=1): L = -log(y_hat)
loss = -np.log(y_hat)

# Gradient of loss wrt w: dL/dw = y_hat - y
grad = y_hat - y

plt.figure(figsize=(12,5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(w, loss, label='Cross-Entropy Loss')
plt.xlabel('Weight w')
plt.ylabel('Loss')
plt.title('Cross-Entropy Loss vs Weight')
plt.grid(True)
plt.legend()

# Plot gradient
plt.subplot(1, 2, 2)
plt.plot(w, grad, label='Gradient dL/dw', color='orange')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Weight w')
plt.ylabel('Gradient')
plt.title('Gradient of Cross-Entropy Loss vs Weight')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
