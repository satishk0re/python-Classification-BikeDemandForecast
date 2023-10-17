"""
This script performs Principal Component Analysis (PCA) on a dataset containing
independent variables to reduce dimensionality and identify the number of principal
components to retain based on a specified variance threshold.
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Load data from a CSV file
data = pd.read_csv(r'C:\Users\satis\python-Classification-BikeDemandForecast\mode_choice_pcasample (1).csv')

# Select independent variables for PCA
independent_variables = data[['travel_time_number', 'age', 'individual_travel_frequency', 'household_vehicles', 'household_size']]

# Standardize the independent variables
scaler = StandardScaler()
independent_variables_scaled = scaler.fit_transform(independent_variables)

# Perform PCA
pca = PCA()
principal_components = pca.fit_transform(independent_variables_scaled)

# Calculate explained variance for each principal component
explained_variance = pca.explained_variance_ratio_

# Calculate Cumulative variance explained
cumulative_variance = explained_variance.cumsum()

# Determine the number of principal components to retain based on a threshold
threshold = 0.95  # Set your desired variance threshold
num_components = sum(cumulative_variance <= threshold) + 1

# Print the results
print("Variance explained by each principal component:")
for i, var in enumerate(explained_variance):
    print(f"Principal Component {i+1}: {var:.4f}")

print("\nCumulative variance explained:")
for i, var in enumerate(cumulative_variance):
    print(f"Principal Components {i+1}: {var:.4f}")

print(f"\nNumber of principal components to retain: {num_components}")
