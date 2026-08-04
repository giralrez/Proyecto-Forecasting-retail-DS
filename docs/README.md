# 🏆 Forecasting de Ventas para Retail de Artículos Deportivos

> **Proyecto integral de Machine Learning para la predicción de ventas utilizando HistGradientBoostingRegressor y una aplicación interactiva desarrollada con Streamlit.**

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Análisis%20de%20Datos-black?style=for-the-badge&logo=pandas)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-ff4b4b?style=for-the-badge&logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-Visualización-3f4f75?style=for-the-badge&logo=plotly)

---

# 📖 Descripción del Proyecto

La predicción de la demanda es uno de los principales desafíos del sector retail, ya que impacta directamente la gestión del inventario, las compras, la logística y la rentabilidad del negocio.

Este proyecto desarrolla una **solución End-to-End de Machine Learning** capaz de pronosticar las ventas de una empresa dedicada a la comercialización de artículos deportivos utilizando **HistGradientBoostingRegressor** con features extendidas (temporales, lags, calendario, competencia).

El proyecto abarca todo el ciclo de vida de un modelo de Machine Learning, desde la preparación de los datos hasta su despliegue en una aplicación web interactiva con predicciones recursivas día por día.

---

# 🎯 Objetivo de Negocio

Construir un modelo predictivo que permita estimar las ventas futuras con el fin de apoyar la toma de decisiones relacionadas con:

* 📦 Gestión de inventarios.
* 📈 Planeación de la demanda.
* 🛒 Optimización de compras.
* 💰 Reducción de sobreinventarios y quiebres de stock.
* 📊 Planeación comercial y análisis estratégico.

---

# 🚀 Tecnologías Utilizadas

* Python 3.x
* Pandas
* NumPy
* Scikit-Learn (HistGradientBoostingRegressor)
* Holidays (festivos Colombia)
* Matplotlib / Seaborn
* Plotly (gráficos interactivos)
* Streamlit (aplicación web)
* Joblib (serialización de modelos)

---

# 📂 Estructura del Proyecto

```text
forecasting_ventas/
│
├── app/
│   └── streamlit/
│       └── app.py                 # Aplicación Streamlit con predicciones recursivas
│
├── data/
│   ├── raw/
│   │   ├── entrenamiento/
│   │   │   ├── ventas.csv         # Datos de ventas 2021-2024
│   │   │   └── competencia.csv    # Precios de competencia
│   │   └── inferencia/
│   │       └── ventas_2025_inferencia.csv  # Datos para predicción 2025
│   ├── processed/
│   │   ├── df.csv                 # Dataset procesado extendido (~70 features)
│   │   └── inferencia_df_transformado.csv  # Inferencia transformada
│   └── documentation.md           # Documentación de datos
│
├── docs/
│   ├── README.md                  # Este archivo
│   └── documentation.md           # Documentación técnica completa
│
├── models/
│   └── modelo_final.joblib        # Modelo v2 serializado
│
├── notebooks/
│   ├── entrenamiento.ipynb        # Notebook principal (EDA + features + modelo)
│   └── forecasting.ipynb          # Notebook de inferencia 2025
│
├── requirements.txt
└── README.md
```

---

# 🔬 Flujo de Trabajo del Proyecto

El proyecto sigue un flujo de trabajo orientado a entornos reales de Machine Learning.

```text
Datos Originales (ventas.csv + competencia.csv)
        │
        ▼
Limpieza y Preparación
        │
        ▼
Análisis Exploratorio (EDA)
        │
        ▼
Ingeniería de Características (~70 features)
├─ Variables temporales (mes, trimestre, día)
├─ Festivos Colombia + eventos comerciales
├─ Lag features (1-7 días)
├─ Media móvil de 7 días
├─ Precio competencia y ratio
└─ One-Hot Encoding (productos, categorías)
        │
        ▼
Entrenamiento del Modelo v2 (HistGradientBoostingRegressor)
        │
        ▼
Serialización del Modelo (joblib)
        │
        ▼
Pipeline de Inferencia (forecasting.ipynb)
├─ Mismas features que entrenamiento
├─ Lags desde octubre como base
└─ Predicciones para noviembre 2025
        │
        ▼
Despliegue con Streamlit
├─ Sidebar con controles de simulación
├─ Predicciones recursivas día por día
├─ KPIs y gráficos interactivos
└─ Comparativa de escenarios de competencia
```

---

# 🤖 Modelo de Machine Learning

El modelo predictivo utiliza **HistGradientBoostingRegressor** de Scikit-Learn, una implementación eficiente de Gradient Boosting optimizada para grandes volúmenes de datos.

### Características del Modelo v2

* **~70 features** incluyendo temporales, lags, calendario, competencia y OHE
* **Predicciones recursivas** día por día con actualización de lags
* **R² = 0.8227** en validación (2024)
* **MAE = 1.51 unidades** en validación
* **Parámetros conservadores** para evitar overfitting

### Ventajas

* Alto poder predictivo en datos tabulares
* Excelente capacidad de generalización
* Manejo eficiente de relaciones no lineales
* Robustez frente a grandes volúmenes de datos
* Manejo nativo de valores faltantes

---

# 📊 Funcionalidades

El proyecto incluye:

✅ Análisis Exploratorio de Datos (EDA).

✅ Limpieza y transformación de datos.

✅ Ingeniería de características extendidas (~70 features).

✅ Variables temporales, lags y calendario.

✅ Entrenamiento del modelo de Machine Learning (HistGradientBoostingRegressor).

✅ Evaluación mediante métricas de desempeño.

✅ Serialización del modelo entrenado (joblib).

✅ Pipeline de inferencia para predicción 2025.

✅ Predicciones recursivas día por día.

✅ Aplicación web interactiva con Streamlit.

✅ Simulación de escenarios de competencia (±5%).

✅ Análisis de impacto de descuentos.

---

# 💻 Ejecución del Proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu_usuario/forecasting_ventas.git
```

### 2. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecutar notebook de entrenamiento

```bash
jupyter notebook notebooks/entrenamiento.ipynb
```

### 4. Ejecutar notebook de inferencia

```bash
jupyter notebook notebooks/forecasting.ipynb
```

### 5. Ejecutar la aplicación Streamlit

```bash
cd app/streamlit
streamlit run app.py
```

---

# 📈 Resultados Esperados

La aplicación permite:

* Seleccionar un producto de la lista de 24 productos.
* Ajustar el descuento sobre el precio base (-50% a +50%).
* Simular 3 escenarios de competencia (Actual, -5%, +5%).
* Visualizar predicciones diarias de ventas para noviembre 2025.
* Identificar visualmente el impacto del Black Friday.
* Comparar unidades totales e ingresos por escenario.
* Descargar resultados detallados por día.

---

# 🖥️ Vista de la Aplicación

> Próximamente se incorporarán capturas de pantalla y un GIF demostrativo del funcionamiento de la aplicación.

* Página principal.
* Panel de predicciones.
* Resultados del modelo.
* Visualización de métricas.

---

# 💼 Competencias Demostradas

Este proyecto evidencia conocimientos en:

* Ciencia de Datos.
* Machine Learning.
* Forecasting de Ventas.
* Ingeniería de Características.
* Modelado Predictivo.
* Análisis Exploratorio de Datos.
* Desarrollo de aplicaciones con Streamlit.
* Organización de proyectos de Machine Learning.
* Buenas prácticas de desarrollo en Python.

---

# 🔮 Mejoras Futuras

Como parte de la evolución del proyecto se contempla incorporar:

* Optimización automática de hiperparámetros (Grid Search / Random Search).
* Explicabilidad del modelo mediante SHAP.
* Validación cruzada temporal (Time Series Cross Validation).
* Seguimiento de experimentos con MLflow.
* Contenerización utilizando Docker.
* API REST con FastAPI.
* Despliegue en la nube (Azure, AWS o Google Cloud).
* Automatización mediante pipelines de CI/CD.
* Monitoreo de data drift y re-entrenamiento programado.

---

# 👤 Autor

Andrés Giraldo Ramírez (@giralrez) Software Engineer en transición hacia Data Analytics, ML y Data Engineering.
---

# ⭐ Apoya este proyecto

Si este proyecto te resultó interesante o te fue útil, considera darle una ⭐ al repositorio. Tu apoyo motiva el desarrollo de nuevas soluciones de Ciencia de Datos y Machine Learning.
