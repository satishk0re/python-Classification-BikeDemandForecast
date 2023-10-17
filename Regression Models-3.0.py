"""
Description: This code loads a bike rental dataset, preprocesses the data, and builds three regression models
(Linear Regression, Random Forest Regression, and Gradient Boosting Regression) to predict bike rental demand.
The code calculates and displays the Mean Squared Error (MSE) for each model to assess their performance.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv(r"C:\Users\satis\python-Classification-BikeDemandForecast\bike (1).csv")

# Convert the "day" column to datetime format
data['day'] = pd.to_datetime(data['day'], format='%d-%m-%Y')

# Extract relevant features and target variable
features = ['season', 'year', 'month', 'hour', 'holiday', 'weekday', 'working_day', 'weather_situation', 'temprature', 'absolute_temprature', 'humidity', 'windspeed']
target = 'count'
X = data[features]
y = data[target]

# Convert categorical variables to one-hot encoding
X = pd.get_dummies(X, columns=['season', 'year', 'month', 'hour', 'holiday', 'weekday', 'weather_situation'], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Linear Regression
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)
prediction_model1 = linear_regression_model.predict(X_test)

# Model 2: Random Forest Regression
forest_regression_model = RandomForestRegressor()
forest_regression_model.fit(X_train, y_train)
prediction_model2 = forest_regression_model.predict(X_test)

# Model 3: Gradient Boosting Regression
gradient_regression_model = GradientBoostingRegressor()
gradient_regression_model.fit(X_train, y_train)
prediction_model3 = gradient_regression_model.predict(X_test)

# Calculate the MSE for each model
mse_model1 = mean_squared_error(y_test, prediction_model1)
mse_model2 = mean_squared_error(y_test, prediction_model2)
mse_model3 = mean_squared_error(y_test, prediction_model3)

# Print the performance metrics
print("Model 1 - Linear Regression:")
print("MSE:", mse_model1)
print()

print("Model 2 - Random Forest Regression:")
print("MSE:", mse_model2)
print()

print("Model 3 - Gradient Boosting Regression:")
print("MSE:", mse_model3)
