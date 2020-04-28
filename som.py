import numpy as np
import matplotlib as plt
import pandas as pd

dataset = pd.read_csv("Credit_Card_Applications.csv")

# Splitting the dataset

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling of dataset before training
# Normalization  Used first in RNN 

from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler(feature_range = (0, 1))

# Now we neet to fit this ms to X, so that all the value
# in X lies between 0 and 1
X = ms.fit_transform(X)

# Training the SOM (MiniSom 1.0)
from minisom import MiniSom

# Dimensions of SOM, not must be too small, 
# for outlier we need a larger matrix, but we have very less customers
# so we take dimensions = 10X10

# learning_rate is how fast the circles in the SOM converge
# if they learn fast, the size of the circles will reduce quickly
# Som object creation
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)

# Train this SOM object to X, but before that we need to Initialize the weights randomly to small nos close to 0

som.random_weights_init(X)

# Train the object on x
# Parameters = data, no of iterations
som.train_random(X, 100)

# Plot the SOM to detect the outlier
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()

# Red circles for customers who didn't get approval
# Green square for the customers who got the approval

# 'o' denotes a circle and 's' denotes a square
markers = ['o', 's']
colors = ['r', 'g']

# i represents all the customers i.e the row in the dataset
# x denotes the whole vector which includes all the columns of a row
# so x[0] denotes the entire row 1, all the columns i.e details of the first customer
# x[1] denotes the entire row 2, i.e. all the columns of the second row
for i, x in enumerate(X):
    # winning node for each customer = w
    w = som.winner(x)
    #w[0] is the first coordinate of winning node, and w[1] is second
    # 0.5 to add row and column wise so as to plot the marker at the center
    plot(w[0] + 0.5, 
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# This gives the dictionary, with keys as the winning nodes and values as the list of all the
# customers belonging to that winning node
#mappings = som.win_map(X)

# from our SOM we look, which cell is an outlier and get the list of customers belonging to that coordingate
#frauds = np.concatenate[mappings(), ]
    
    
