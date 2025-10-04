import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


"""
XOR gate implementation using Cross Entropy Loss
For a network with a hidden layer
"""


def train(x, y, W1, W2, b1, b2, learning_rate=0.2, epochs=100000):
    for epoch in range(epochs):
        # forward pass
        z1 = np.dot(x, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        # checking loss
        epsilon = 1e-8
        loss = -np.mean(y * np.log(a2 + epsilon) + (1 - y) * np.log(1 - a2 + epsilon))

        if loss < 0.005:
            break

        delta_2 = a2 - y
        dW2 = np.dot(a1.T, delta_2) / len(x)
        db2 = np.sum(delta_2) / len(x)
        da1 = a1 * (1 - a1)
        delta_1 = np.dot(delta_2, W2.T) * da1  # backpropagation
        dW1 = np.dot(x.T, delta_1) / len(x)
        db1 = np.sum(delta_1) / len(x)

        # updating weighs and biases
        W2 -= learning_rate * dW2
        W1 -= learning_rate * dW1
        b2 -= learning_rate * db2
        b1 -= learning_rate * db1
    print(f"Epoch {epoch}: Loss = {loss:.6f}")
    return W1, W2, b1, b2


def predict(X_test, W1, W2, b1, b2):
    z1 = np.dot(X_test, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    predictions = (a2 > 0.5).astype(int)
    return a2, predictions

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0]).reshape(-1, 1)
n_in_W1, n_out_W1, n_in_W2, n_out_W2 = 2, 2, 2, 1
limit_W1, limit_W2 = np.sqrt(6 / (n_in_W1 + n_out_W1)), np.sqrt(
    6 / (n_in_W2 + n_out_W2)
)
W1 = np.random.uniform(low=-limit_W1, high=limit_W1, size=(n_in_W1, n_out_W1))
W2 = np.random.uniform(low=-limit_W2, high=limit_W2, size=(n_in_W2, n_out_W2))
b1 = np.array([0.1, 0.1])
b2 = np.array([0.1])
learning_rate = 0.1

W1_final, W2_final, b1_final, b2_final = train(x, y, W1, W2, b1, b2)
probability, result = predict(np.array([1, 1]), W1_final, W2_final, b1_final, b2_final)
print(f"Probability: {probability}, Prediction: {result}")