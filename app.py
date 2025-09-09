from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# ---------------------------------------------
# Función para generar gráfica matplotlib en base64
# ---------------------------------------------
def generar_grafica(metrics):
    fig, ax = plt.subplots(figsize=(6,4))
    nombres = list(metrics.keys())
    valores = list(metrics.values())
    ax.bar(nombres, valores, color=["#007bff", "#28a745", "#ffc107", "#dc3545"])
    ax.set_title("Resultados del modelo")
    ax.set_ylim(0, 1)  # porque las métricas van de 0 a 1
    ax.set_ylabel("Valor")
    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format="png")
    img.seek(0)
    grafica_b64 = base64.b64encode(img.getvalue()).decode("utf-8")
    plt.close(fig)
    return grafica_b64


# ---------------------------------------------
# Página principal
# ---------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------
# Entrenamiento del modelo
# ---------------------------------------------
from sklearn.metrics import classification_report, confusion_matrix

@app.route("/train", methods=["GET", "POST"])
def train():
    if request.method == "POST":
        file = request.files["dataset"]
        algoritmo = request.form["algoritmo"]
        validacion = request.form["validacion"]
        parametro = int(request.form.get("parametro", 5))

        # Guardar dataset
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        data = pd.read_csv(filepath)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Selección de modelo
        if algoritmo == "id3":
            model = DecisionTreeClassifier(criterion="entropy")
        elif algoritmo == "j48":
            ccp_alpha = float(request.form.get("ccp_alpha", 0.0))
            model = DecisionTreeClassifier(criterion="entropy", ccp_alpha=ccp_alpha)
        elif algoritmo == "knn":
            model = KNeighborsClassifier(n_neighbors=parametro)
        else:
            return "Algoritmo no válido"

        metrics = {}
        report_html = None
        cm_b64 = None

        if validacion == "holdout":
            test_size = parametro / 100
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Métricas
            metrics["Accuracy"] = accuracy_score(y_test, y_pred)
            metrics["Precision"] = precision_score(y_test, y_pred, average="macro", zero_division=0)
            metrics["Recall"] = recall_score(y_test, y_pred, average="macro", zero_division=0)
            metrics["F1"] = f1_score(y_test, y_pred, average="macro", zero_division=0)

            # Reporte de clasificación
            report_html = "<pre>" + classification_report(y_test, y_pred, zero_division=0) + "</pre>"

            # Matriz de confusión
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=model.classes_, yticklabels=model.classes_)
            plt.xlabel("Predicción")
            plt.ylabel("Real")
            img = io.BytesIO()
            plt.tight_layout()
            plt.savefig(img, format="png")
            img.seek(0)
            cm_b64 = base64.b64encode(img.getvalue()).decode("utf-8")
            plt.close(fig)

        elif validacion == "kfold":
            scores = cross_val_score(model, X, y, cv=parametro, scoring="accuracy")
            model.fit(X, y)
            metrics["Accuracy"] = scores.mean()
            metrics["Precision"] = 0.0
            metrics["Recall"] = 0.0
            metrics["F1"] = 0.0
            # 🚨 en K-Fold no hay y_test/y_pred → no generamos reporte ni matriz

        # Guardar modelo
        model_name = f"{algoritmo}_model.joblib"
        joblib.dump(model, os.path.join(MODEL_FOLDER, model_name))

        # Generar gráfica de métricas
        grafica = generar_grafica(metrics)

        # 🚨 SIN tocar el árbol
        arbol_b64 = None
        if algoritmo in ["id3", "j48"]:
            fig, ax = plt.subplots(figsize=(12, 8))
            plot_tree(model, filled=True, feature_names=X.columns,
                      class_names=[str(c) for c in model.classes_])
            img = io.BytesIO()
            plt.savefig(img, format="png")
            img.seek(0)
            arbol_b64 = base64.b64encode(img.getvalue()).decode("utf-8")
            plt.close(fig)

        model_info = {
            "model": model,
            "columns": list(X.columns)
        }

        model_name = f"{algoritmo}_model.joblib"
        joblib.dump(model_info, os.path.join(MODEL_FOLDER, model_name))

        return render_template(
            "results.html",
            metrics=metrics,
            model_name=model_name,
            grafica=grafica,
            arbol_b64=arbol_b64,
            report=report_html,
            cm_b64=cm_b64
        )

    return render_template("train.html")



# ---------------------------------------------
# Clasificación con modelo guardado
# ---------------------------------------------
@app.route("/classify", methods=["GET", "POST"])
def classify():
    if request.method == "POST":
        model_name = request.form["modelo"]
        model_info = joblib.load(os.path.join(MODEL_FOLDER, model_name))
        model = model_info["model"]
        columnas = model_info["columns"]

        # Opción 1: CSV
        if "dataset" in request.files and request.files["dataset"].filename != "":
            file = request.files["dataset"]
            data = pd.read_csv(file)
            preds = model.predict(data)
            resultados = pd.DataFrame({"Entrada": data.index, "Predicción": preds})
            return render_template(
                "classify.html",
                modelos=os.listdir(MODEL_FOLDER),
                resultado_tabla=resultados.to_html(classes="table table-bordered"),
                columnas=columnas
            )

        # Opción 2: Inputs individuales
        else:
            valores = [float(request.form.get(col)) for col in columnas]
            prediccion = model.predict([valores])[0]
            return render_template(
                "classify.html",
                modelos=os.listdir(MODEL_FOLDER),
                resultado=prediccion,
                columnas=columnas
            )

    # GET → mostrar formulario
    if os.listdir(MODEL_FOLDER):
        model_info = joblib.load(os.path.join(MODEL_FOLDER, os.listdir(MODEL_FOLDER)[0]))
        columnas = model_info["columns"]
    else:
        columnas = []

    return render_template("classify.html", modelos=os.listdir(MODEL_FOLDER), columnas=columnas)

if __name__ == "__main__":
    app.run(debug=True)
