# Proyecto de Predicción de Iris

Este proyecto utiliza el famoso conjunto de datos Iris para construir un modelo de clasificación que predice la especie de iris basándose en las características de la flor.

## Estructura del Proyecto

```
/IrisPredictionProject
│
├── README.md                  # Este archivo
├── requirements.txt           # Dependencias del proyecto
├── iris_prediction.ipynb      # Notebook con el análisis y modelo
├── informe_resultados.pdf     # Informe detallado de resultados
├── /images
│   └── arbol_decision.png     # Visualización del modelo de árbol de decisión
└── /data
    └── iris.csv               # Conjunto de datos
```

## Requisitos

Para ejecutar este proyecto, necesitarás Python 3.7+ y las bibliotecas listadas en `requirements.txt`. Para instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Uso

1. Abre el notebook `iris_prediction.ipynb` en Jupyter:
   ```bash
   jupyter notebook iris_prediction.ipynb
   ```

2. Ejecuta todas las celdas para:
   - Cargar y explorar los datos
   - Preprocesar los datos
   - Entrenar varios modelos
   - Evaluar el rendimiento
   - Visualizar los resultados

3. Consulta `informe_resultados.pdf` para un análisis detallado de los hallazgos.

## Conjunto de Datos

El conjunto de datos Iris contiene 150 muestras de flores de iris, divididas en tres especies: setosa, versicolor y virginica. Para cada muestra, se midieron cuatro características:

- Longitud del sépalo (cm)
- Ancho del sépalo (cm)
- Longitud del pétalo (cm)
- Ancho del pétalo (cm)

## Resultados

El modelo final logra una precisión del X% en el conjunto de prueba. Los detalles completos se pueden encontrar en el notebook y en el informe de resultados.