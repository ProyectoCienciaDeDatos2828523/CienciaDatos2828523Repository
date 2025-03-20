import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Crear el conjunto de datos
data = {
    'Empleado': ['Ana', 'Luis', 'Carlos'],
    'Salud': [6, 8, 7],
    'Finanzas': [7, 6, 8],
    'Relaciones': [8, 7, 6]
}

df = pd.DataFrame(data)

# 2. Calcular estadísticas
mean_salud = np.mean(df['Salud'])
mean_finanzas = np.mean(df['Finanzas'])
mean_relaciones = np.mean(df['Relaciones'])

print(f"Media de Salud: {mean_salud}")
print(f"Media de Finanzas: {mean_finanzas}")
print(f"Media de Relaciones: {mean_relaciones}")

# 3. Visualizar los resultados con un gráfico de barras
areas = ['Salud', 'Finanzas', 'Relaciones']
medias = [mean_salud, mean_finanzas, mean_relaciones]

plt.bar(areas, medias, color='orange')
plt.title('Satisfacción Promedio en Diferentes Áreas')
plt.xlabel('Área')
plt.ylabel('Satisfacción Promedio')
plt.ylim(0, 10)  # Escala de satisfacción de 0 a 10
plt.show()


