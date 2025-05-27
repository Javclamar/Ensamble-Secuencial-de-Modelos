# Ensamble Secuencial De Modelos Predictivos

## Descripción
Este proyecto implementa un meta-algoritmo de ensamble secuencial para tareas de regresión (y potencialmente clasificación), siguiendo una estrategia donde cada modelo intenta corregir los errores del anterior, 
inspirándose en técnicas como el Gradient Boosting. La implementación se ha realizado desde cero usando Python y scikit-learn, y se ha diseñado para ser modular, extensible y fácilmente aplicable a distintos conjuntos de datos.

##Objetivos

- Desarrollar un ensamble secuencial de modelos predictivos (meta-modelo) desde cero.
- Aplicarlo a tareas de regresión sobre diferentes conjuntos de datos.
- Evaluar el comportamiento del algoritmo frente a diferentes configuraciones de hiperparámetros.
- Comparar el rendimiento del meta-modelo frente a modelos base individuales.

##Fundamentos del Algoritmo

El algoritmo se basa en una estrategia aditiva y secuencial de aprendizaje:

1. Inicializa una predicción base (por ejemplo, la media de y).ç

2.En cada iteración:
- Calcula los residuos o gradientes del error respecto a la predicción actual.
- Entrena un modelo base sobre esos residuos.
- Actualiza la predicción final con una fracción del resultado del nuevo modelo (tasa de aprendizaje).

3. En tareas de clasificación, la predicción continua se convierte en una clase mediante un umbral.

##Conjunto de datos

Se han utilizado dos conjuntos de datos proporcionados por la asignatura:

  Dataset 1: Precio de viviendas
  Dataset 2: Datos sobre la enfermedad de Parkinson

Ambos conjuntos han sido preprocesados adecuadamente y convertidos a formato numérico cuando ha sido necesario.

## Uso del Código

- meta_model.py: Contiene la implementación del meta-modelo (entrenamiento, predicción).

- experiments.ipynb: Cuaderno Jupyter con los experimentos, visualizaciones y resultados.

- data/: Carpeta con los datasets utilizados (en formato .csv).

- README.md: Este archivo.

##Ejecución

1. Abrir el archivo experiments.ipynb.
2.Ejecutar las celdas desde el inicio. El código está documentado y se puede ajustar fácilmente para nuevos datasets.
3. Se puede modificar el estimador base (DecisionTreeRegressor, LinearRegression, etc.) y sus hiperparámetros.

##Requisitos

- Python 3.8+
- Scikit-learn
- Numpy
- Pandas
- Matplotlib / Seaborn (para visualización)
