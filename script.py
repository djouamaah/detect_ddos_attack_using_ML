https://aseados.ucd.ie/datasets/SDN/

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# Load dataset
data = pd.read_csv("InSDN_dataset.csv")

# Preprocessing
# ... (data cleaning, normalization, etc.)

# Clustering
kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(data)

# Classification
classifier = RandomForestClassifier()
classifier.fit(data, clusters)

# Evaluation
predictions = classifier.predict(data)
accuracy = accuracy_score(clusters, predictions)
f1 = f1_score(clusters, predictions)
recall = recall_score(clusters, predictions)
precision = precision_score(clusters, predictions)

print("Accuracy: ", accuracy)
print("F1-measure: ", f1)
print("Recall: ", recall)
print("Precision: ", precision)
