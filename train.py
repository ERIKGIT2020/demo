import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ==========================================
# Configuración de MLflow
# ==========================================

# Usar SQLite como backend (compatible con GitHub Actions)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Crear o seleccionar experimento
mlflow.set_experiment("Diabetes")


# ==========================================
# Cargar datos
# ==========================================

df = pd.read_csv("data/diabetes.csv")


# ==========================================
# Separar variables
# ==========================================

X = df.drop(["PatientID", "Diabetic"], axis=1)
y = df["Diabetic"]


# ==========================================
# División entrenamiento / prueba
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# Entrenamiento con MLflow
# ==========================================

with mlflow.start_run():

    # Crear modelo
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )


    # Entrenar
    model.fit(
        X_train,
        y_train
    )


    # Predicción
    y_pred = model.predict(X_test)


    # Evaluación
    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    print("----------------------------")
    print("Modelo entrenado")
    print(f"Accuracy: {accuracy:.4f}")
    print("----------------------------")


    # ======================================
    # Registrar parámetros
    # ======================================

    mlflow.log_param(
        "Modelo",
        "RandomForestClassifier"
    )

    mlflow.log_param(
        "n_estimators",
        100
    )

    mlflow.log_param(
        "random_state",
        42
    )


    # ======================================
    # Registrar métrica
    # ======================================

    mlflow.log_metric(
        "Accuracy",
        accuracy
    )


    # ======================================
    # Guardar modelo en MLflow
    # ======================================

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="modelo"
    )


# ==========================================
# Fin
# ==========================================

print("Entrenamiento terminado correctamente")



