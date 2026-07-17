import pandas as pd
import joblib

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

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Crear el modelo
model = RandomForestClassifier(random_state=42)

# Entrenar
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluar
accuracy = accuracy_score(y_test, y_pred)

print("=" * 40)
print("Modelo entrenado correctamente")
print(f"Accuracy: {accuracy:.2%}")
print("=" * 40)

# Guardar modelo
joblib.dump(model, "modelo_diabetes.pkl")

print("Modelo guardado como modelo_diabetes.pkl")