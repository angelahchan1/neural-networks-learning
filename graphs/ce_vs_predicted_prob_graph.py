import numpy as np
import matplotlib.pyplot as plt

# Create range of predicted probabilities from 0.001 to 0.999
p = np.linspace(0.001, 0.999, 200)

# Cross-entropy loss when the true label is 1
loss_y1 = -np.log(p)

# Cross-entropy loss when the true label is 0
loss_y0 = -np.log(1 - p)

# Plotting
plt.figure(figsize=(10, 6))

# Plot loss curves
plt.plot(p, loss_y1, label='True Label = 1', color='blue')
plt.plot(p, loss_y0, label='True Label = 0', color='red')

# Vertical line at 0.5 probability (decision threshold)
plt.axvline(0.5, color='gray', linestyle='--', label='Prediction = 0.5')

plt.xlabel('Predicted Probability')
plt.ylabel('Cross-Entropy Loss')
plt.title('Cross-Entropy Loss vs Predicted Probability')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
