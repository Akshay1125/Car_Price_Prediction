# Importing the libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn import metrics

os.chdir(r'C:\Users\Admin\python projects\NEW PROJECTS\archive (2)')
# Importing the dataset
dataset = pd.read_csv('car data.csv')
X = dataset.drop(['Car_Name','Selling_Price'], axis=1)
y = dataset.iloc[:, 2:3].values
print(X)
print(y)

# one hot encoding
# Encoding categorical data
X = pd.get_dummies(X, columns=['Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)
print(X)
print("Shape of X is: ", X.shape)
print("Shape of y is: ", y.shape)

# train,test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training the Multiple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(-1,1) ,y_test.reshape(-1,1) ), 1))

print("MAE: ", (metrics.mean_absolute_error(y_pred, y_test)))
print("MSE: ", (metrics.mean_squared_error(y_pred, y_test)))
print("R2 score: ", (metrics.r2_score(y_pred, y_test)))

# present price vs selling price 
# fig=plt.figure(figsize=(7,5))
# plt.title('Correlation between present price and selling price')
# sns.regplot(x='Present_Price', y='Selling_Price',data=dataset)

sns.regplot(x=y_pred, y=y_test, scatter_kws={"color": "green"}, line_kws={"color": "black"})
plt.xlabel("Predicted Price")
plt.ylabel('Actual Price')
plt.title("ACtual vs predicted price")
plt.show()
