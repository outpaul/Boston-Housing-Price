import pandas as pd
import matplotlib.pyplot as plt
#import tkinter as tk
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

boston = load_boston()
lr = LinearRegression()

b = pd.DataFrame(boston.data)

lr.fit(b, boston.target)

#Add any other dataset for this model to predict in place of b
print(lr.predict(b)[:1])

# An ideal plot would be an x = y graph
plt.scatter(lr.predict(b), boston.target)
plt.title('Prediction vs Real Housing Price')
plt.xlabel('Predicted Housing Price')
plt.ylabel('Real Housing Price')
plt.show()
