---
name: forecasting-ventas-notebook
user-invocable: true
description: "Guía paso a paso para crear y mejorar notebooks de análisis y modelado de forecasting de ventas con enfoque en limpieza, visualización, ingeniería de variables y preparación para modelos."
---

# Forecasting Ventas Notebook

## Objetivo

Esta skill guía a un desarrollador de datos o científico de datos a través de un flujo estructurado para construir notebooks de forecasting de ventas en el proyecto `forecasting_ventas`.

## Cuándo usar

- Cuando necesitas crear o refactorizar un notebook de análisis de ventas.
- Cuando buscas una secuencia clara para preparar datos, explorar patrones y generar features para modelos de forecasting.
- Cuando el notebook debe seguir las reglas de variables y librerías del proyecto.

## Flujo recomendado

1. Revisar los datos de origen y cargar los archivos relevantes.
2. Verificar tipos de datos, fechas y valores faltantes.
3. Unir datos de ventas y competencia en un solo dataframe limpio.
4. Crear variables temporales y de negocio relevantes para forecasting.
5. Visualizar patrones clave: tendencias por fecha, categoría, producto y competencia.
6. Preparar datos para modelado con lags, medias móviles y codificación adecuada.
7. Guardar resultados procesados para uso en entrenamiento e inferencia.

## Estilo y restricciones

- Usa español en los comentarios y descripciones de las celdas.
- Sigue la convención de trazabilidad en cada celda: `# Trazabilidad XX - ...`.
- Aplica sólo las librerías permitidas: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyter`, `streamlit`, `holidays`.
- No uses variables diferentes a las listadas en `.github/copilot-instructions.md`, salvo variables que definas tú mismo en el notebook.

## Ejemplo de prompts

- `Crea un notebook paso a paso para preparar datos de forecasting de ventas con análisis exploratorio y generación de features.`
- `Refactoriza este notebook para que use trazabilidad de celdas y solo las variables permitidas por el proyecto.`
- `Agrega una sección de visualización de competencia comparando precio propio versus Amazon y Decathlon.`

## Resultado esperado

- Notebook estructurado con celdas claras, cada una con su comentario de trazabilidad.
- Flujo reproducible que combina carga, validación de datos, análisis exploratorio y preparación para modelado.
- Cumplimiento de las restricciones del proyecto sobre variables y librerías.
