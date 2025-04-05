# 📚 Importación de bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# 🔍 Cargar el dataset Iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="target")

# 📊 Ver los primeros registros
print("Primeros datos:")
print(X.head())
print("Etiquetas:")
print(y.head())

# 📈 Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 🌳 Crear y entrenar el modelo de Árbol de Decisión
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)

# 🤖 Realizar predicciones
y_pred = modelo.predict(X_test)

# 🧮 Evaluación del modelo
print("\n📌 Precisión del modelo:")
print(accuracy_score(y_test, y_pred))

print("\n📌 Reporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 🌳 Visualizar el árbol de decisión
plt.figure(figsize=(12, 8))
plot_tree(
    modelo,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True
)
plt.title("Árbol de Decisión para Clasificación de Iris")
plt.savefig("images/arbol_decision.png")  # Guarda la imagen para el informe
plt.show()
