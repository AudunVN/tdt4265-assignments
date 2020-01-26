import mnist
import numpy as np
import matplotlib.pyplot as plt

BIAS = 1

def sigmoid(z):
    return 1. / (1 + np.exp(-z))

def z(weights, x):
    assert weights.shape[0] == x.shape[1]  # weights should have as many rows as x has features.
    return np.dot(x, weights)

def hypothesis(weights, x):
    return sigmoid(z(weights, x))

def cost(weights, x, y):
    assert x.shape[1] == weights.shape[0]  # x has a column for each feature, weights has a row for each feature.
    assert x.shape[0] == y.shape[0]  # One row per sample.
    h = hypothesis(weights, x)
    one_case = np.matmul(-y.T, np.log(h))
    zero_case = np.matmul(-(1 - y).T, np.log(1 - h))
    return (one_case + zero_case) / len(x)

def gradient_descent(weights, x, y, learning_rate, regularization = 0):
    regularization = weights * regularization
    error = hypothesis(weights, x) - y
    n = (learning_rate / len(x)) * (np.matmul(x.T, error) + regularization)
    return weights - n

def minimize(weights, x, y, iterations, learning_rate, regularization = 0):
    costs = []
    for _ in range(iterations):
        weights = gradient_descent(weights, x, y, learning_rate, regularization)
        costs.append(cost(weights, x, y))
    return weights, costs

X_train, Y_train, X_test, Y_test = mnist.load()
X_train_vec = np.array(X_train)
Y_train_vec = np.array(Y_train)
X_test_vec = np.array(X_test)
Y_test_vec = np.array(Y_test)

print(X_train_vec.shape[1])

weights = np.array([0] * X_train_vec.shape[1])
weights, costs = minimize(weights, X_train_vec, Y_train_vec, 50, 0.001, 1)
plt.plot(range(len(costs)), costs)
plt.show()
print(costs[-1])

predictions = X_test_vec.dot(weights) > 0

print(predictions)