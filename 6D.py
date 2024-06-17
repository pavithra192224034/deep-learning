import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv("/content/IRIS.csv")  # Make sure the path is correct
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Decision Tree Classification model on the Training set
classifier = DecisionTreeClassifier(criterion='entropy', random_state=5)
classifier.fit(X_train, y_train)

# Display the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(classifier, filled=True, rounded=True, feature_names=dataset.columns[:-1])
plt.show()

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Display the results (confusion matrix and accuracy)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
