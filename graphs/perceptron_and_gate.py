import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Data: AND gate
# -----------------------------
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]], dtype=float)
y = np.array([0, 0, 0, 1], dtype=float)

# -----------------------------
# Model: single perceptron with sigmoid
# -----------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def d_sigmoid(z):
    s = sigmoid(z)
    return s * (1.0 - s)

def forward(X, w, b):
    z = X @ w + b
    a = sigmoid(z)
    return z, a

# -----------------------------
# Training loop (batch gradient descent)
# -----------------------------
np.random.seed(0)
w = np.random.randn(2) * 0.5
b = 0.0

lr = 0.5         # learning rate
epochs = 120     # total epochs

# Record history for animation
w_hist = []
b_hist = []
loss_hist = []

for ep in range(epochs):
    z, a = forward(X, w, b)
    # Mean squared error loss
    loss = 0.5 * np.mean((a - y)**2)
    loss_hist.append(loss)

    # Backprop (batch)
    # dL/da = (a - y), d a/dz = d_sigmoid(z), so dL/dz = (a - y) * d_sigmoid(z)
    delta = (a - y) * d_sigmoid(z)  # shape (4,)
    # dL/dw = X^T @ delta / N (average across samples)
    grad_w = (X.T @ delta) / len(X)
    grad_b = np.mean(delta)

    # Update
    w -= lr * grad_w
    b -= lr * grad_b

    w_hist.append(w.copy())
    b_hist.append(b)

w_hist = np.array(w_hist)  # shape (epochs, 2)
b_hist = np.array(b_hist)  # shape (epochs,)

# -----------------------------
# Animation setup
# -----------------------------
fig, ax = plt.subplots(figsize=(6,6))

# Plot data points
# "Don't fire" points in blue, "Fire" point in red
non_fire = X[y == 0]
fire = X[y == 1]
ax.scatter(non_fire[:,0], non_fire[:,1], c='tab:blue', marker='x', s=100, label="Don't fire (0)")
ax.scatter(fire[:,0], fire[:,1], c='tab:red', marker='o', s=100, label='Fire (1)')

ax.set_xlim(-0.25, 1.25)
ax.set_ylim(-0.25, 1.25)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Perceptron learning AND: Decision boundary over epochs')
ax.legend(loc='upper left')

# Decision boundary line (updated each frame)
line, = ax.plot([], [], color='tab:green', lw=2)
epoch_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# Helper to compute boundary points from w, b
def decision_boundary_points(w, b, x_range=(-0.25, 1.25)):
    x1 = np.linspace(x_range[0], x_range[1], 200)
    # If w2 is near zero, draw a vertical line x = -b/w1
    if np.abs(w[1]) < 1e-6:
        x_v = -b / w[0] if np.abs(w[0]) > 1e-6 else np.nan
        return np.array([x_v, x_v]), np.array([x_range[0], x_range[1]])
    # Otherwise y = -(w1/w2) x - b/w2
    x2 = -(w[0]/w[1]) * x1 - b / w[1]
    return x1, x2

def init():
    line.set_data([], [])
    epoch_text.set_text('')
    return line, epoch_text

def update(frame):
    w_curr = w_hist[frame]
    b_curr = b_hist[frame]
    x1, x2 = decision_boundary_points(w_curr, b_curr)
    line.set_data(x1, x2)
    epoch_text.set_text(f'Epoch: {frame+1}/{epochs}\nLoss: {loss_hist[frame]:.4f}')
    return line, epoch_text

anim = FuncAnimation(fig, update, frames=len(w_hist), init_func=init, interval=60, blit=True)

plt.show()

# Optional: save animation (uncomment one)
# anim.save('perceptron_and.gif', writer='pillow', fps=30)   # requires pillow
# anim.save('perceptron_and.mp4', fps=30)                    # requires ffmpeg
