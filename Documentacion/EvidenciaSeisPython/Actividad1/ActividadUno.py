import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

# Carga el conjunto de datos Iris desde seaborn
data = sns.load_dataset('iris')

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

# Convierte las etiquetas de 'species' en valores numéricos con LabelEncoder
label_encoder = LabelEncoder()
df['species'] = label_encoder.fit_transform(df['species'])

# Ver las primeras filas del conjunto de datos
print(data.head())
print(data.describe())

# Generar boxplots
plt.figure(figsize=(12, 8))
sns.boxplot(data=data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
plt.title("Boxplot de las características del Iris")
plt.savefig("boxplot_iris.png")
plt.show()

# Histogramas de las características numéricas
data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].hist(bins=15, figsize=(12, 8))
plt.suptitle("Histogramas de las características del Iris")
plt.savefig("histogramas_iris.png")
plt.show()

# Pairplot para ver la relación entre las características numéricas
sns.pairplot(data, hue='species')
plt.suptitle("Pairplot de las características del Iris", y=1.02)
plt.savefig("pairplot_iris.png")
plt.show()

# Matriz de correlación
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Mapa de calor de la correlación entre características")
plt.savefig("heatmap_iris.png")
plt.show()

# Gráfico de caja para comparar la longitud del sépalo entre las especies
plt.figure(figsize=(8, 6))
sns.boxplot(x='species', y='sepal_length', data=data)
plt.title("Comparación de la longitud del sépalo por especie")
plt.savefig("boxplot_sepal_length.png")
plt.show()
