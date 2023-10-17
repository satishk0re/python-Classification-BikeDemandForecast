# Bike Rental Prediction Data Preprocessing

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv(r"C:\Users\satis\python-Classification-BikeDemandForecast\bike (1).csv")

# Data Preprocessing
data = data.drop(['instant', 'casual', 'registered'], axis=1)

# Convert categorical variables into dummy variables
categorical_vars = ['season', 'year', 'month', 'hour', 'holiday', 'weekday', 'weather_situation']
data = pd.get_dummies(data, columns=categorical_vars)

# Normalize numeric variables
numeric_vars = ['temprature', 'absolute_temprature', 'humidity', 'windspeed']
scaler = StandardScaler()
data[numeric_vars] = scaler.fit_transform(data[numeric_vars])

# Split the dataset into training and testing subsets
X = data.drop("count", axis=1)
y = data["count"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
