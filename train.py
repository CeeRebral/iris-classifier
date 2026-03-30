from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

kneighbors = KNeighborsClassifier(n_neighbors=3)

iris = load_iris()
x = iris.data # shape (150,4)
y = iris.target # shape (150,)
print(iris.feature_names, iris.target_names)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("predictions:", y_pred[:5])
print("actual labels:", y_test[:5]

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


model2 = KNeighborsClassifier (n_neighbors=5)
model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test)
print("k-NN accuracy:", accuracy_score(y_test, y_pred2))

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(x_train, y_train)
y_pred3 = model3.predict(X_test)
print ("example of overfitting", accuracy_score(y_test, y_pred))