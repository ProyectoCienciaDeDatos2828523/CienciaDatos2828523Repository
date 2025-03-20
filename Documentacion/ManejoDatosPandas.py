import pandas as pd

# Crear un diccionario con los datos de empleados
data = {
    'Empleado': ['Ana', 'Luis', 'Carlos'],
    'Satisfacción': [7, 8, 5],
    'Áreas de mejora': ['Comunicación', 'Gestión del tiempo', 'Liderazgo']
}

# Crear un DataFrame a partir del diccionario
df = pd.DataFrame(data)

# Mostrar el DataFrame completo
print("DataFrame original:")
print(df)

# Filtrar los empleados con satisfacción mayor a 6
df_filtrado = df[df['Satisfacción'] > 6]
print("\nEmpleados con satisfacción mayor a 6:")
print(df_filtrado)

# Calcular el promedio de satisfacción
mean_satisfaction = df['Satisfacción'].mean()
print(f"\nPromedio de satisfacción: {mean_satisfaction}")

import pandas as pd
print("Pandas funciona correctamente")

