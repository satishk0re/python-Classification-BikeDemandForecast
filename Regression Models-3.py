# Bike Rental Demand Prediction using Regression Models

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
data = pd.read_csv(r"C:\Users\satis\python-Classification-BikeDemandForecast\bike (1).csv")

# Convert the "day" column to datetime format
data['day'] = pd.to_datetime(data['day'], format="%d-%m-%Y")

# Extract relevant features and target variable
features = ['season', 'year', 'month', 'hour', 'holiday', 'weekday', 'working_day', 'weather_situation', 'temprature', 'absolute_temprature', 'humidity', 'windspeed']
target = 'count'
X = data[features]
y = data[target]

# Convert categorical variables to one-hot encoding
X = pd.get_dummies(X, columns=['season', 'month', 'hour', 'weekday', 'weather_situation'], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training and Evaluation
# Model 1: Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
prediction_model1 = linear_model.predict(X_test)

# Model 2: Random Forest Regression
random_forest_model = RandomForestRegressor()
random_forest_model.fit(X_train, y_train)
prediction_model2 = random_forest_model.predict(X_test)

# Model 3: Gradient Boosting Regression
gradient_boosting_model = GradientBoostingRegressor()
gradient_boosting_model.fit(X_train, y_train)
prediction_model3 = gradient_boosting_model.predict(X_test)

# Model Evaluation
# Calculate the Mean Squared Error (MSE) for each model
mse_model1 = mean_squared_error(y_test, prediction_model1)
mse_model2 = mean_squared_error(y_test, prediction_model2)
mse_model3 = mean_squared_error(y_test, prediction_model3)

# Combine predicted_labels using Voting Ensemble
voting_predictions = np.round((prediction_model1 + prediction_model2 + prediction_model3) / 3)

# Evaluate the ensemble model
ensemble_mse = mean_squared_error(y_test, voting_predictions)
print("Ensemble Model Mean Squared Error:", ensemble_mse)
