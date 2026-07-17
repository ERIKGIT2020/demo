import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ==========================================
# Configuración MLflow
# ==========================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Registro de modelos
mlflow.set_registry_uri("sqlite:///mlflow.db")

# Crear experimento
mlflow.set_experiment("Diabetes")


# ==========================================
# Cargar datos
# ==========================================

df = pd.read_csv("data/diabetes.csv")


# ==========================================
# Variables
# ==========================================

X = df.drop(
    ["PatientID", "Diabetic"],
    axis=1
)

y = df["Diabetic"]


# ==========================================
# Separar datos
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# Entrenamiento MLflow
# ==========================================

with mlflow.start_run():


    # Crear modelo

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )


    # Entrenar

    model.fit(
        X_train,
        y_train
    )


    # Predicción

    y_pred = model.predict(
        X_test
    )


    # Accuracy

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
        200
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
    # Registrar modelo en MLflow Registry
    # ======================================

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="modelo",
        registered_model_name="Diabetes_Model",
        input_example=X_train.iloc[:5]
    )


print("Entrenamiento terminado correctamente")


