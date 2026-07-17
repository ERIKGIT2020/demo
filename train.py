import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ==========================
# Leer datos
# ==========================
df = pd.read_csv("data/diabetes.csv")


# Variables de entrada y salida
X = df.drop(["PatientID", "Diabetic"], axis=1)
y = df["Diabetic"]


# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Crear modelo
model = RandomForestClassifier(
    random_state=42
)


# ==========================
# MLflow Tracking
# ==========================

with mlflow.start_run():

    # Entrenamiento
    model.fit(X_train, y_train)


    # Predicción
    y_pred = model.predict(X_test)


    # Métrica
    accuracy = accuracy_score(y_test, y_pred)


    print("=" * 40)
    print("Modelo entrenado correctamente")
    print(f"Accuracy: {accuracy:.2%}")
    print("=" * 40)


    # Guardar información en MLflow
    mlflow.log_param(
        "Modelo",
        "RandomForest"
    )

    mlflow.log_param(
        "n_estimators",
        model.n_estimators
    )


    mlflow.log_metric(
        "Accuracy",
        accuracy
    )


    # Guardar modelo en MLflow
    mlflow.sklearn.log_model(
        model,
        "modelo"
    )


print("Proceso terminado")