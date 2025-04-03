import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Cargar dataset de seaborn (usar un solo dataset)
data = sns.load_dataset('iris')

# Seleccionamos las características y variable objetivo
x = data[['sepal_length', 'sepal_width', 'petal_width']]  # Variables independientes
y = data['petal_length']  # Variable objetivo

# Dividir en conjunto de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Verificar tamaños de los conjuntos
print(f"Tamaño del conjunto de entrenamiento: {x_train.shape}, {y_train.shape}")
print(f"Tamaño del conjunto de prueba: {x_test.shape}, {y_test.shape}")

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(x_train, y_train)

# Ver coeficientes e intercepto
print(f"Coeficientes: {model.coef_}")
print(f"Intersección (intercepto): {model.intercept_}")

# Realizar predicciones
y_pred = model.predict(x_test)

# Mostrar predicciones
predictions_df = pd.DataFrame({'Real': y_test.values, 'Predicción': y_pred})
print(predictions_df.head())

# Calcular MSE y R²
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Error cuadrático medio (MSE): {mse}")
print(f"Coeficiente de determinación R²: {r2}")

# Graficar resultados
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicciones')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label="Línea de referencia")
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.title("Valores reales vs Predicciones")
plt.legend()
plt.show()