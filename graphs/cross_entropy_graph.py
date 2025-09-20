import numpy as np
import matplotlib.pyplot as plt

# Predicted probabilities from 0 to 1
p = np.linspace(0.001, 0.999, 100)

# Cross-entropy loss when true label y=1
loss_for_1 = -np.log(p)

# Cross-entropy loss when true label y=0
loss_for_0 = -np.log(1 - p)

plt.plot(p, loss_for_1, label='True label = 1')
plt.plot(p, loss_for_0, label='True label = 0')
plt.xlabel('Predicted Probability')
plt.ylabel('Cross-Entropy Loss')
plt.title('Cross-Entropy Loss vs Predicted Probability')
plt.legend()
plt.grid(True)
plt.show()
