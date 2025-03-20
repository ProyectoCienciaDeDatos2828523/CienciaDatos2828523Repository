import numpy as np

def main():
    # Crear un array de NumPy con 5 números enteros
    array = np.array([10, 20, 30, 40, 50])
    
    # Calcular la suma de todos los elementos del array
    sum_array = np.sum(array)
    
    # Calcular la media del array
    mean_array = np.mean(array)
    
    # Obtener la desviación estándar del array
    std_array = np.std(array)
    
    # Imprimir los resultados
    print(f"Suma: {sum_array}, Media: {mean_array}, Desviación estándar: {std_array}")
    
    # Reflexión sobre los resultados
    print("\nReflexión:")
    print("La suma de los elementos puede indicar el nivel total de satisfacción.")
    print("La media proporciona una idea del nivel promedio de satisfacción.")
    print("La desviación estándar muestra la variabilidad de las respuestas.")

if __name__ == "__main__":
    main()