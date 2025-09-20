import numpy as np
import matplotlib.pyplot as plt
'''
AND gate implementation using Cross Entropy Loss
For a single perceptron
'''
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def d_sigmoid(output):
    return output * (1 - output)

x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0,0,0,1])
weights = np.array([6,6], dtype=float)
bias = -9.0
learning_rate = 0.1

def train_perceptron(x, y, weights, bias, learning_rate=0.1, epochs=300000):
    for epoch in range(epochs):
        z = np.dot(x, weights) + bias
        predictions = sigmoid(z)
        epsilon = 1e-8
        loss = -np.mean(y * np.log(predictions + epsilon) + (1 - y) * np.log(1 - predictions + epsilon))

        if epoch % 10000 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f}")

        if loss < 0.01:
            break

        delta = predictions - y
        weight_gradients = np.dot(x.T, delta)
        mean_gradients = weight_gradients / len(x)
        weights -= learning_rate * mean_gradients
        bias -= learning_rate * np.sum(delta)/ len(x)
    return weights, bias

final_weights, final_bias = train_perceptron(x, y, weights, 0.1, 0.1)

def calculate_predictions (weights, bias):
    z = np.dot(x, weights) + bias
    predictions = sigmoid(z)
    return predictions

print("\nFinal Predictions:", calculate_predictions(final_weights, final_bias))
print("Final Weights:", final_weights)
print("Final Bias:", final_bias)









