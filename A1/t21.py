import mnist
import numpy as np
X_train, Y_train, X_test, Y_test = mnist.load()

INPUT_X_RES = 28
INPUT_Y_RES = 28
ALPHA = 0.001 # learning rate
N_TRAINING_SAMPLES = 100
N_ITERATIONS = 1000

def predict(features, weights):
  z = np.dot(features, weights)
  return sigmoid(z)

def cost_function(w, y):
    return (-y * np.log(w) - (1 - y) * np.log(1 - w)).mean()

def update_weights(features, labels, weights, alpha):
    
    N = len(features)

    predictions = predict(features, weights)

    gradient = np.dot(features.T, predictions-labels)

    gradient /= N

    weights -= alpha * gradient

    return weights

def train(features, labels, weights, alpha, iterations):
    cost_history = []

    for i in range(iters):
        weights = update_weights(features, labels, weights, lr)

        #Calculate error for auditing purposes
        cost = cost_function(features, labels, weights)
        cost_history.append(cost)

        if i % 1000 == 0:
            print("iter: "+str(i) + " cost: "+str(cost))

    return weights, cost_history
    
def decide(prob):
    return 1 if prob >= .5 else 0

def classify(predictions):
    decisions = np.vectorize(decide)
    return decisions(predictions).flatten()