# Linear Support Vector Machine (SVM) for Binary Classification

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Define the dataset
data_points = np.array([[-1.0, 1.0],
                        [0.0, 1.0],
                        [0.0, 0.5],
                        [1.0, 1.0],
                        [-2.0, 1.0],
                        [-1.5, -1.0],
                        [1.5, -1.0],
                        [2.0, 1.0]])

labels = np.array([1, 1, 1, 1, -1, -1, -1, -1])

# Train a linear SVM model
svm = SVC(kernel='linear')
svm.fit(data_points, labels)

# Get support vectors and coefficients of the decision boundary
support_vectors = svm.support_vectors_
coefficients = svm.coef_[0]
intercept = svm.intercept_

# Plot the decision boundary and support vectors
plt.scatter(data_points[:, 0], data_points[:, 1], c=labels)
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], color='red', marker='x', label='Support Vectors')
x = np.linspace(-3, 3, 100)
y_boundary = (-intercept - coefficients[0] * x) / coefficients[1]
plt.plot(x, y_boundary, color='black', label='Decision Boundary')

plt.xlabel('data_x1')
plt.ylabel('data_x2')
plt.title('Linear SVM Decision Boundary')
plt.legend()
plt.show()
