# Support Vector Machine (SVM) for Binary Classification

import numpy as np
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
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(data_points, labels)

# New samples to predict
new_samples = np.array([[2.0, -1.0],
                        [1.0, 0.0],
                        [-0.5, 0.5],
                        [1.0, 2.0],
                        [0.0, -1.0]])

# Predict labels for new samples
predicted_labels = svm_classifier.predict(new_samples)

# Print the predicted labels
for sample, prediction in zip(new_samples, predicted_labels):
    print(f"Sample {sample} is predicted as class {prediction}")
