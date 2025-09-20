import numpy as np
import matplotlib.pyplot as plt

# Your trained weights and bias from before:
weights = final_weights
bias = final_bias

# AND gate inputs and labels:
x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])

# Plot points with color based on label
plt.scatter(x[y==0][:,0], x[y==0][:,1], color='red', label='Class 0')
plt.scatter(x[y==1][:,0], x[y==1][:,1], color='blue', label='Class 1')

# Plot decision boundary: w1*x1 + w2*x2 + b = 0
# Solve for x2: x2 = -(w1*x1 + b) / w2
x1_vals = np.linspace(-0.5, 1.5, 100)
x2_vals = -(weights[0] * x1_vals + bias) / weights[1]

plt.plot(x1_vals, x2_vals, 'g-', label='Decision Boundary')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('AND Gate Decision Boundary')
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.legend()
plt.grid(True)
plt.show()
