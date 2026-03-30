from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
x = iris.data # shape (150,4)
y = iris.target # shape (150,)
print(iris.feature_names, iris.target_names)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Setup Sample Data (Actual values vs Predicted values)
# 0 = Negative Class, 1 = Positive Class
actual    = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
predicted = [0, 1, 1, 1, 0, 0, 1, 0, 1, 1]

# 2. Compute the confusion matrix
cm = confusion_matrix(actual, predicted)

# 3. Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification


X, y = make_classification(n_samples=100, n_features=10, random_state=42)
model = LogisticRegression()
model.fit(X, y)


filename = 'my_trained_model.joblib'
joblib.dump(model, filename)

print(f"Model saved to {filename}")

import joblib
# Save the model to a file
joblib.dump(model, 'my_model.joblib')

# Load the model from the file
loaded_model = joblib.load('my_model.joblib')
