import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Para asegurarse de que la ventana de Matplotlib se abra

# Crear los datos
data = {
    'Empleado': ['Ana', 'Luis', 'Carlos'],
    'Satisfacción': [7, 8, 5],
    'Áreas de mejora': ['Comunicación', 'Gestión del tiempo', 'Liderazgo']
}

# Crear DataFrame
df = pd.DataFrame(data)

# Gráfico de barras
plt.figure(figsize=(8,5))
plt.bar(df['Empleado'], df['Satisfacción'], color='blue')
plt.title('Nivel de Satisfacción de los Empleados')
plt.xlabel('Empleado')
plt.ylabel('Satisfacción')
plt.ylim(0, 10)
plt.show()

# Gráfico de líneas
plt.figure(figsize=(8,5))
plt.plot(df['Empleado'], df['Satisfacción'], marker='o', color='green', linestyle='dashed')
plt.title('Satisfacción de los Empleados a lo largo del tiempo')
plt.xlabel('Empleado')
plt.ylabel('Satisfacción')
plt.ylim(0, 10)
plt.show()