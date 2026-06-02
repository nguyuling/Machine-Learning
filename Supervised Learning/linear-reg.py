import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load data
path = 'Data/Salary_Data.csv'
headernames = ['year-of-experience', 'salary']
dataset = pd.read_csv(path, names=headernames)
print(dataset.head(3))

# define feature and target
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model training
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# model testing
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual Values': y_test, 'Predicted Values': y_pred})
print(df)

# model performance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print('MSE: ', mean_squared_error(y_test, y_pred))
print('MAE: ', mean_absolute_error(y_test, y_pred))
print('R2 score: ', r2_score(y_test, y_pred))

# scatter plot of train test data
plt.scatter(X, y, color='green')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary (INR)')
# plt.show()