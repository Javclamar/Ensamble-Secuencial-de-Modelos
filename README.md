# 📊 Ensamble Secuencial de Modelos Predictivos  

## 📌 Descripción  

Este proyecto implementa un meta-algoritmo de aprendizaje automático basado en la técnica de ensamble secuencial de modelos predictivos. La idea es combinar varios modelos (en este caso regresores) entrenados de forma secuencial,
de manera que cada nuevo modelo se especializa en corregir los errores (residuos) cometidos por los modelos anteriores.  

Se trata de un enfoque inspirado en métodos como el Gradient Boosting, adaptado y desarrollado desde cero para comprender su funcionamiento interno y experimentar con diferentes configuraciones.  

## ⚙️ Tecnologías utilizadas  

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)  
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)  
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)  
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)  
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)  

## 📦 Estructura del proyecto
bash
````
/data               # Conjuntos de datos de experimentación
/notebooks          # Cuadernos Jupyter con código y experimentos
/docs               # Documentos de guía y memoria
README.md           # Este archivo
````

## 🧩 Implementación
El proyecto se basa en los siguientes pasos:  

- Inicializar una predicción base (media de la variable objetivo o cero).

- Entrenar secuencialmente varios modelos, cada uno sobre los residuos (errores) del meta-modelo en ese momento.

- Calcular la predicción final como la suma de las predicciones de todos los modelos ponderadas por la tasa de aprendizaje.

## 📈 Conjuntos de datos utilizados  

- Datos sobre precio de viviendas, con valor objetivo el precio de las viviendas

- Datos sobre la enfermedad de Parkinson, con valor objetivo el índice de probabilidad

## 🧪 Experimentación  

- Validación cruzada para evaluar el rendimiento medio.

- Comparación con modelos simples (baseline).

- Exploración de hiperparámetros para analizar el impacto en el rendimiento.

- Implementación de early stopping para detener el entrenamiento cuando no haya mejora significativa.

## ✏️ Autores
Trabajo desarrollado en el marco de la asignatura Inteligencia Artificial (IS) 2024/25 port Javier Clavijo Martínez y Manuel Roberto López Pavía
