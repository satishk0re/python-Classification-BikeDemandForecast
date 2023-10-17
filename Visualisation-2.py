# Simple Data Visualization

import matplotlib.pyplot as plt

# Define the data points and their labels
data = [
        [-1.0, 1.0, 1],  # (x, data_y, label)
        [0.0, 1.0, 1],
        [0.0, 0.5, 1],
        [1.0, 1.0, 1],
        [-2.0, 1.0, -1],
        [-1.5, -1.0, -1],
        [1.5, -1.0, -1],
        [2.0, 1.0, -1]
]

# Separate data points by class
class_1_points = [point[:2] for point in data if point[2] == 1]
class_minus_1_points = [point[:2] for point in data if point[2] == -1]

# Plotting the data points
plt.scatter(*zip(*class_1_points), label='Class 1', marker='o', color='blue')
plt.scatter(*zip(*class_minus_1_points), label='Class -1', marker='x', color='red')

# Add legend to the plot
plt.legend()

# Display the plot
plt.show()
