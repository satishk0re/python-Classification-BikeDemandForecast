# Support Vector Machine (SVM) for Binary Classification

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

# Define the data points and their labels
data_x1 = [-1.0, 0.0, 0.0, 1.0, -2.0, -1.5, 1.5, 2.0]
data_x2 = [1.0, 1.0, 0.5, 1.0, 1.0, -1.0, -1.0, 1.0]
data_y = [1, 1, 1, 1, -1, -1, -1, -1]

# Plot the scatter plot
plt.scatter(data_x1, data_x2, c=data_y)
plt.xlabel('data_x1')
plt.ylabel('data_x2')
plt.title('Dataset')
plt.show()

# Convert data to NumPy arrays for SVM
x1 = np.array([data_x1])
x2 = np.array([data_x2])
y = np.array([data_y])

# Create a meshgrid for visualization
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))

# Fit SVM with RBF kernel
svm = SVC(kernel='rbf', gamma='scale')
svm.fit(np.column_stack((data_x1, data_x2)), data_y)

# Predict on meshgrid points
Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.scatter(data_x1, data_x2, c=data_y)
plt.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='black')
plt.xlabel('data_x1')
plt.ylabel('data_x2')
plt.title('Linearly Separable Dataset')
plt.show()
