import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import tkinter as tk
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

boston = load_boston()

# This linear model uses the whole data as a training set
lr = LinearRegression()

b = pd.DataFrame(boston.data)

lr.fit(b, boston.target)

mse = round(np.mean((lr.predict(b) - boston.target) ** 2),2)

# lr Residual Plot
plt.subplot(2,1,1)
plt.scatter(lr.predict(b),lr.predict(b) - boston.target)
plt.hlines(y = 0, xmin = 0, xmax = 40)
plt.title(' lr Residual Plot')
plt.xlabel('Predicted Price')
plt.ylabel('Error(mse = ' + str(mse) + ')')

# Implementing training set split on another linear model
trainX, testX, trainY, testY = train_test_split(b, boston.target, test_size = 0.3333, random_state = 3)

lr2 = LinearRegression()

lr2.fit(trainX, trainY)

mse2 = round(np.mean((lr.predict(testX) - testY) ** 2),2)

# lr2 Residual Plot
plt.subplot(2,1,2)
plt.scatter(lr2.predict(trainX), lr2.predict(trainX) - trainY, c = 'b', alpha = 1)
plt.scatter(lr2.predict(testX), lr2.predict(testX) - testY, c = 'r')
plt.hlines(y = 0, xmin = 0, xmax = 40)
plt.xlabel('Predicted Price')
plt.ylabel('Error(mse = ' + str(mse2) + ')')
plt.title('lr2 Residual Plot')

# Displaying the plots
plt.show()
