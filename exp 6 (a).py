import numpy as np
import pandas as pd

# Load the dataset
dataset = pd.read_csv("/content/IRIS.csv")

# Display dataset shape and first few rows for verification
print("Dataset shape:", dataset.shape)
print("First few rows of the dataset:")
print(dataset.head())

# Check if the last column is indeed the target variable
print("Columns in the dataset:", dataset.columns)

# Assuming the IRIS dataset has features and labels in the last column
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the K-Nearest Neighbors (K-NN) Classification model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Display the results (confusion matrix and accuracy)
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)
