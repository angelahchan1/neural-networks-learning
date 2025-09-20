import numpy as np
import matplotlib.pyplot as plt

# Predicted probability values from 0.001 to 0.999 (avoid log(0))
p = np.linspace(0.001, 0.999, 500)

# Cross-entropy loss for true label = 1
loss_y1 = -np.log(p)

# Cross-entropy loss for true label = 0
loss_y0 = -np.log(1 - p)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(p, loss_y1, label='True Label = 1 (Loss = -log(p))', color='blue', linewidth=2)
plt.plot(p, loss_y0, label='True Label = 0 (Loss = -log(1 - p))', color='red', linewidth=2)

# Add vertical line at 0.5
plt.axvline(0.5, color='gray', linestyle='--', linewidth=1)
plt.text(0.52, 4, 'Prediction = 0.5', fontsize=10, color='gray')

# Labels and title
plt.title('Binary Cross-Entropy Loss vs Predicted Probability')
plt.xlabel('Predicted Probability of Class 1')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
