import os
import joblib
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay


def train(test_size=0.2, random_state=42):

    # Create outputs folder
    os.makedirs("outputs", exist_ok=True)

    # 1. Load data
    iris = load_iris()
    X = iris.data
    y = iris.target

    print(iris.feature_names, iris.target_names)

    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 3. Train model
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # 4. Predictions
    y_pred = model.predict(X_test)

    print("Predictions:", y_pred[:5])
    print("Actual labels:", y_test[:5])

    # 5. Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # 6. Confusion Matrix
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=iris.target_names
    )

    disp.figure_.savefig("outputs/confusion_matrix.png")
    plt.close()

    # 7. Save model
    joblib.dump(model, "outputs/model.joblib")

    print("All outputs saved in 'outputs/' folder")


if __name__ == "__main__":
    train()