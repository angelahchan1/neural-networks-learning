import numpy as np
import matplotlib.pyplot as plt

# Simulate predicted probabilities (from 0.001 to 0.999)
softmax_outputs = np.linspace(0.001, 0.999, 500)

# Cross-entropy loss = -log(probability of the correct class)
# We'll simulate it for different levels of "certainty" by scaling the predicted prob

loss_high_conf = -np.log(softmax_outputs)           # Model predicted the correct class directly
loss_medium_conf = -np.log(softmax_outputs * 0.8)   # Model assigned 80% of p to correct class
loss_low_conf = -np.log(softmax_outputs * 0.5)      # Model less sure (50%)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(softmax_outputs, loss_high_conf, label='Correct class prob = p', color='blue')
plt.plot(softmax_outputs, loss_medium_conf, label='Correct class prob = 0.8 × p', color='orange')
plt.plot(softmax_outputs, loss_low_conf, label='Correct class prob = 0.5 × p', color='green')

plt.title('Categorical Cross-Entropy Loss vs Predicted Probability of Correct Class')
plt.xlabel('Predicted Probability for Correct Class')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
