import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import calculate_cost_gradient, compute_cost,\
     predict, make_meshgrid, plot_contours, sgd

# We'll use a synthetic data set
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import argparse


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=int, default=5000,
	help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.000001,
	help="learning rate")
ap.add_argument("-r", "--regularization_strength", type=float, default=10000.0,
	help="The hyperparameter C")

args = vars(ap.parse_args())

regularization_strength = args['regularization_strength']
learning_rate = args["alpha"]
max_epochs = args["epochs"]


"""The next code generates a random n-class classification problem. 
Besides, it introduces interdependence between generated features and 
adds various types of further noise to the data. In other words, we're 
getting a synthetic dataset. """

X,Y = make_classification(n_samples=100, n_features=2, n_informative=2,n_redundant=0, 
                          n_repeated=0, n_classes=2, n_clusters_per_class=2, class_sep=2,
                          flip_y=0.05, weights=[0.5,0.5], random_state=5)
# y_i \in \{-1,+1\}
Y = np.array([-1.0 if x == 0 else 1.0 for x in Y])

# Let's normalize data for better convergence:
X_normalized = MinMaxScaler().fit_transform(X)
X = pd.DataFrame(X_normalized)

# insert 1 in every row for intercept b
X.insert(loc=len(X.columns), column='intercept', value=1)

# split data into train and test set
print("splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

# train
print("Training started...")
W, lossHistory = sgd(X_train, y_train, learning_rate, regularization_strength, max_epochs)

print("Training finished.")

# testing
print("Testing...")
y_train_predicted = np.array([])
for i in range(X_train.shape[0]):
    yp = np.sign(np.dot(X_train[i], W))
    y_train_predicted = np.append(y_train_predicted, yp)

y_test_predicted = np.array([])
for i in range(X_test.shape[0]):
    yp = np.sign(np.dot(X_test[i], W))
    y_test_predicted = np.append(y_test_predicted, yp)

print("accuracy on test dataset: {}".format(accuracy_score(y_test, y_test_predicted)))

# Now, let's explore the classification boundaries

fig, ax = plt.subplots()
title = ('Decision surface of linear SVC ')
# Set-up grid for plotting.
X0, X1 = X_test[:, 0], X_test[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, xx, yy, W, cmap=plt.cm.PuBu_r, alpha=1)
ax.scatter(X0, X1, c=y_test, cmap=plt.cm.Set1, s=40, edgecolors=None, marker='o')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
plt.show()


plt.plot(range(1, len(lossHistory) + 1), lossHistory)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('SVM SGD training Loss')
plt.show()