import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# read the data
data = pd.read_csv('challenge_dataset.txt', sep = ',', header = None)
x_values = np.asarray(data.ix[:,0])
y_values = np.asarray(data.ix[:,1])

# train the model
linear_regression = LinearRegression()
linear_regression.fit(x_values.reshape(x_values.shape[0], 1), y_values)
error = np.mean((y_values - linear_regression.predict(x_values.reshape(x_values.shape[0], 1))) ** 2)

# plot the data and the linear regression model
plt.scatter(x_values, y_values)
plt.plot(x_values, linear_regression.predict(x_values.reshape(x_values.shape[0], 1)))
plt.show()
print error