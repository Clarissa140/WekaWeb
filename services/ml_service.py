from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
import numpy as np

def train_model(X, y, algorithm, validation, k=5, test_size=0.3):
    # Seleccionar algoritmo
    if algorithm == "ID3":
        model = DecisionTreeClassifier(criterion="entropy")
    elif algorithm == "KNN":
        model = KNeighborsClassifier(n_neighbors=k)
    else:
        raise ValueError("Algoritmo no soportado")

    # Validación
    if validation == "holdout":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred, average="macro"),
            "F1-Score": f1_score(y_test, y_pred, average="macro")
        }

    elif validation == "kfold":
        scores = cross_val_score(model, X, y, cv=k, scoring="accuracy")
        metrics = {
            "Accuracy": np.mean(scores),
            "Recall": None,  # Se pueden añadir métricas adicionales
            "F1-Score": None
        }
        model.fit(X, y)  # reentrenar con todo
    else:
        raise ValueError("Método de validación no soportado")

    return model, metrics

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "Recall": recall_score(y, y_pred, average="macro"),
        "F1-Score": f1_score(y, y_pred, average="macro")
    }
