import numpy as np
import matplotlib.pyplot as plt

# Predicted probability of the correct class
p = np.linspace(0.001, 0.999, 500)

# Cross-entropy loss: -log(p)
loss = -np.log(p)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(p, loss, label=r'Cross-Entropy Loss: $-\log(p)$', color='purple', linewidth=2)

# Highlight the point at p = 0.1 (10% confidence)
p_point = 0.1
loss_point = -np.log(p_point)
plt.scatter([p_point], [loss_point], color='red', zorder=5)
plt.text(p_point + 0.02, loss_point, f'({p_point:.1f}, {loss_point:.2f})', fontsize=12, color='red')

# Formatting
plt.title('Cross-Entropy Loss vs Predicted Probability for Correct Class')
plt.xlabel('Predicted Probability of Correct Class')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
