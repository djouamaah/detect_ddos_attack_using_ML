import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# Load the InSDN dataset
data = np.load("insdn_dataset.npy")

# Perform k-means clustering on the dataset
kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
labels = kmeans.labels_

# Split the data into training and testing sets
train_data = data[:int(len(data) * 0.8), :]
train_labels = labels[:int(len(labels) * 0.8)]
test_data = data[int(len(data) * 0.8):, :]
test_labels = labels[int(len(labels) * 0.8):]

# Train a SVM classifier on the training data
clf = SVC(kernel='linear', C=1, random_state=0)
clf.fit(train_data, train_labels)

# Make predictions on the testing data
predictions = clf.predict(test_data)

# Evaluate the system using various metrics
acc = accuracy_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions)
recall = recall_score(test_labels, predictions)
precision = precision_score(test_labels, predictions)
print("Accuracy: ", acc)
print("F1-measure: ", f1)
print("Recall: ", recall)
print("Precision: ", precision)

