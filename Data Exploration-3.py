# Bike Rental Data Exploration and Analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Load the dataset
data = pd.read_csv(r"C:\Users\satis\python-Classification-BikeDemandForecast\bike (1).csv")

# Data Exploration
print(data.head())

# Summary statistics of numeric variables
print(data.describe())

# Distribution of the target variable 'count'
plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='count', kde=True)
plt.title("Distribution of Rental Bike Demand")
plt.xlabel("Rental Bike Demand (count)")
plt.ylabel("Count")
plt.show()

# Exclude non-numeric columns for correlation calculation
numeric_data = data.select_dtypes(include=['int64', 'float64'])

# Relationship between numeric variables using correlation matrix
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Analyzing non-numeric variables
categorical_data = data.select_dtypes(include=['object'])

# Bar plot of the 'season' variable
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='season')
plt.title("Distribution of Seasons")
plt.xlabel("Season")
plt.ylabel("Count")
plt.show()

# Grouped summary statistics of season and rental bike demand
grouped_data = data.groupby('season')['cnt'].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(data=grouped_data, x='season', y='cnt')
plt.title("Mean Rental Bike Demand by Season")
plt.xlabel("Season")
plt.ylabel("Average Rental Bike Demand (cnt)")
plt.show()

# Chi-square test between 'season' and rental bike demand
contingency_table = pd.crosstab(data['season'], data['cnt'])
chi2, p_value, _, _ = chi2_contingency(contingency_table)
print("Chi-square p-value:", p_value)
