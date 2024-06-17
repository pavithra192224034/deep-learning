import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Importing the dataset
"""
After importing the necessary libraries, we read the dataset. 
Ensure the path to your dataset ('/content/IRIS.csv') is correct.
"""
dataset = pd.read_csv("/content/IRIS.csv")

# Splitting the dataset into features (X) and target variable (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)

# Feature Scaling
"""
Feature scaling is applied to standardize the features. 
StandardScaler is used to scale the training and test sets.
"""
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Logistic Regression (LR) Classification model on the Training set
"""
Logistic Regression (LR) classifier is instantiated and trained on the scaled training data.
"""
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Display the results (confusion matrix and accuracy)
"""
Evaluation metrics such as confusion matrix and accuracy are computed to evaluate the performance 
of the Logistic Regression model.
"""
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
