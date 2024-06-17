# Import the necessary libraries
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the digits dataset
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)

# Train the model
clf = RandomForestClassifier(random_state=23)
clf.fit(X_train, y_train)

# Prediction
y_pred = clf.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
sns.heatmap(cm,
            annot=True,
            fmt='g',
            xticklabels=range(10),
            yticklabels=range(10))
plt.ylabel('Actual', fontsize=13)
plt.xlabel('Predicted', fontsize=13)
plt.title('Confusion Matrix', fontsize=17)
plt.show()

# Finding accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy   :", accuracy)

# Since it's a multiclass classification, use the macro average for precision, recall, and F1-score
precision = precision_score(y_test, y_pred, average='macro')
print("Precision :", precision)
recall = recall_score(y_test, y_pred, average='macro')
print("Recall    :", recall)
f1 = f1_score(y_test, y_pred, average='macro')
print("F1-score  :", f1)
