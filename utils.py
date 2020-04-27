import numpy as np
from sklearn.utils import shuffle

"""The following code calculate the cost based on hinge loss. Note the intercept term $b$ is missing, 
since we'll add an extra column on $X$ with all 1s before splitting our dataset"""

def compute_cost(W, X, Y, regularization_strength):
    N = X.shape[0]    
    distances = np.maximum(0, (1 - Y * (np.dot(X, W))))
    hinge_loss = regularization_strength * (np.sum(distances))
    cost = 1 / 2 * np.dot(W, W) + hinge_loss
    return cost

def calculate_cost_gradient(W, X_batch, Y_batch, regularization_strength):
    if type(Y_batch) == np.float64:
        Y_batch = np.array([Y_batch])
        # gives multidimensional array
        X_batch = np.array([X_batch])

    distance = 1 - (Y_batch * np.dot(X_batch, W))
    dw = np.zeros(len(W))

    for ind, d in enumerate(distance):
        # J(\mathbf{w}) = \frac{\partial}{\partial\mathbf{w}}J(\mathbf{w})
        # = \mathbf{w} \text{ if } \text{max}(0, 1-y_i\mathbf{w^Tx_i}) = 0
        if max(0, d) == 0:
            di = W
        else:
            # \mathbf{w} - Cy_i\mathbf{x_i}
            di = W - (regularization_strength * Y_batch[ind] * X_batch[ind])
        dw += di

    dw = dw/len(Y_batch)  # average
    return dw


def sgd(features, outputs, learning_rate, regularization_strength, max_epochs):    
    weights = np.zeros(features.shape[1])
    nth = 0
    prev_cost = float("inf")
    cost_threshold = 0.01  # in percent
    # SGD
    lossHistory = []

    for epoch in range(1, max_epochs):
        epochLoss = []

        # shuffle to prevent repeating update cycles
        X, Y = shuffle(features, outputs)
        for ind, x in enumerate(X):
            ascent = calculate_cost_gradient(weights, x, Y[ind], regularization_strength)
            weights = weights - (learning_rate * ascent)                    
            
            cost = compute_cost(weights, features, outputs, regularization_strength)
            epochLoss.append(cost)

        lossHistory.append(np.average(epochLoss))

        # convergence check on 2^nth epoch
        if epoch == 2 ** nth or epoch == max_epochs - 1:            
            # cost = compute_cost(weights, features, outputs, regularization_strength)                        
            print("Epoch is: {} and Cost is: {}".format(epoch, cost))
            
            # criterion for stop
            if abs(prev_cost - cost) < cost_threshold * prev_cost:
                return weights, lossHistory
            prev_cost = cost
            nth += 1

    return weights, lossHistory

def predict(X_test, W):
    y_test_predicted = np.array([])
    for i in range(X_test.shape[0]):
        yp = np.sign(np.dot(X_test[i], W))
        y_test_predicted = np.append(y_test_predicted, yp)
    
    return y_test_predicted

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, xx, yy, sgd, **params):
    x_to_predict = np.c_[xx.ravel(), yy.ravel()]
    x_to_predict = np.insert(x_to_predict, 2, 1, axis=1)
    Z = predict(x_to_predict, sgd)
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out