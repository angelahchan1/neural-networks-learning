# Incorrect and doesn't work as AND gate is classification not regression
# fixed by swapping MSE for CE
import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def d_sigmoid(output):
    return output * (1 - output)

x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0,0,0,1])
weights = np.array([6,6])
bias = -9
learning_rate = 0.1

def train_perceptron(x, y, weights, bias, learning_rate=0.1, epochs=300000):
    for epoch in range(epochs):
        z = np.dot(x, weights) + bias
        predictions = sigmoid(z)
        errors = y - predictions
        mse = np.sum(errors*errors) / (2 * len (x))
        if (mse < 0.1):
            break
        delta = errors * d_sigmoid(z) 
        weight_gradients = np.dot(x.T, delta)
        mean_gradients = weight_gradients / len(x)
        weights = weights - (learning_rate * mean_gradients)
        bias = bias - (learning_rate * np.sum(delta)/ len(x))
    return weights, bias

final_weights, final_bias = train_perceptron(x, y, weights, 0.1, 0.1)

def calculate_predictions (weights, bias):
    z = np.dot(x, weights) + bias
    predictions = sigmoid(z)
    return predictions

print(calculate_predictions(final_weights,final_bias))

calculate_predictions(weights, bias)
print(final_weights, final_bias)










