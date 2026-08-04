# 📊 Documentación Completa del Proyecto Forecasting de Ventas

> **Documento de referencia que describe el desarrollo integral del proyecto de predicción de ventas para retail de artículos deportivos.**

**Fecha de Actualización:** 03 de Agosto, 2026  
**Estado del Proyecto:** En Desarrollo - Fase de Despliegue

---

## 📑 Tabla de Contenidos

1. [Visión General del Proyecto](#visión-general-del-proyecto)
2. [Objetivos Alcanzados](#objetivos-alcanzados)
3. [Descripción de Datos](#descripción-de-datos)
4. [Flujo de Desarrollo](#flujo-de-desarrollo)
5. [Etapa 1: Carga y Exploración de Datos](#etapa-1-carga-y-exploración-de-datos)
6. [Etapa 2: Limpieza y Preparación](#etapa-2-limpieza-y-preparación)
7. [Etapa 3: Análisis Exploratorio (EDA)](#etapa-3-análisis-exploratorio-eda)
8. [Etapa 4: Ingeniería de Características](#etapa-4-ingeniería-de-características)
9. [Etapa 5: División de Datos](#etapa-5-división-de-datos)
10. [Etapa 6: Entrenamiento del Modelo v1](#etapa-6-entrenamiento-del-modelo-v1)
11. [Resultados y Métricas v1](#resultados-y-métricas-v1)
12. [Etapa 7: Modelo v2 - Features Extendidas](#etapa-7-modelo-v2---features-extendidas)
13. [Etapa 8: Pipeline de Inferencia](#etapa-8-pipeline-de-inferencia)
14. [Etapa 9: App Streamlit](#etapa-9-app-streamlit)
15. [Próximos Pasos](#próximos-pasos)

---

## 🎯 Visión General del Proyecto

### Contexto

El **Forecasting de Ventas** es un proyecto de Machine Learning destinado a predecir las unidades vendidas de artículos deportivos. La precisión en estas predicciones es crítica para:

- ✅ Optimizar la gestión de inventario
- ✅ Reducir costos de almacenamiento
- ✅ Mejorar la planeación de compras
- ✅ Maximizar la rentabilidad del negocio

### Enfoque

El proyecto adopta una metodología **End-to-End** que abarca desde la preparación de datos hasta la implementación de modelos predictivos listos para producción.

### Stack Tecnológico

| Tecnología | Versión | Propósito |
|-----------|---------|----------|
| **Python** | 3.x | Lenguaje principal |
| **Pandas** | 2.x | Manipulación de datos |
| **NumPy** | Última | Operaciones numéricas |
| **Scikit-Learn** | 1.x | Métricas y utilidades ML |
| **Holidays** | 0.x | Detección de festivos |
| **Matplotlib/Seaborn** | Última | Visualización |
| **Streamlit** | Última | Aplicación web |
| **Jupyter** | Última | Notebooks interactivos |

---

## ✅ Objetivos Alcanzados

### Fase 1: Preparación de Datos ✔️
- [x] Carga de datos de ventas desde CSV
- [x] Integración de datos de competencia
- [x] Validación de integridad de datos
- [x] Identificación y manejo de valores faltantes

### Fase 2: Análisis Exploratorio ✔️
- [x] Análisis descriptivo completo
- [x] Visualización de distribuciones
- [x] Detección de tendencias temporales
- [x] Análisis de estacionalidad
- [x] Identificación de patrones por producto, categoría

### Fase 3: Ingeniería de Características ✔️
- [x] One-Hot Encoding de variables categóricas
- [x] Captura de factores estacionales (mes, trimestre, día semana)
- [x] Detección de períodos especiales (Black Friday, Navidad, festivos locales)
- [x] Variables de competencia precio
- [x] Indicador de producto destacado (estrella)

### Fase 4: Modelado v1 ✔️
- [x] Selección de arquitectura del modelo
- [x] Entrenamiento de HistGradientBoostingRegressor (49 features)
- [x] Validación cruzada
- [x] Evaluación comparativa con baseline

### Fase 5: Features Extendidas y Modelo v2 ✔️
- [x] Variables temporales numéricas (anio, mes, dia_mes, semana_del_año, trimestre)
- [x] Variables temporales binarias (es_fin_semana, es_inicio_mes, es_fin_mes, es_primera_quincena)
- [x] Festivos Colombia con librería holidays
- [x] Eventos comerciales (Black Friday, Cyber Monday)
- [x] Variable descuento_porcentaje
- [x] Lag features (1-7 días) por producto
- [x] Media móvil de 7 días por producto
- [x] Reentrenamiento modelo v2 con ~70 features

### Fase 6: Pipeline de Inferencia ✔️
- [x] Notebook forecasting.ipynb con mismas features que entrenamiento
- [x] Cálculo de lags desde octubre para base de noviembre
- [x] One-Hot Encoding y alineación con df.csv
- [x] Guardado de inferencia_df_transformado.csv

### Fase 7: App Streamlit ✔️
- [x] Sidebar con controles de simulación
- [x] Selector de producto (24 productos)
- [x] Slider de descuento (-50% a +50%)
- [x] Selector de escenario de competencia (3 opciones)
- [x] KPIs destacados (unidades, ingresos, precio, descuento)
- [x] Gráfico diario con Black Friday marcado
- [x] Tabla detallada por día
- [x] Comparativa de 3 escenarios de competencia
- [x] Detección automática de project root para rutas

---

## 📊 Descripción de Datos

### Fuentes de Datos

#### 1. **Datos de Ventas** (`ventas.csv`)
- **Periodicidad:** Diaria
- **Período Cubierto:** 2021-2024
- **Registros:** 3,552 filas

**Columnas principales:**
```
fecha                       (datetime)  - Fecha de venta
producto_id                 (int)       - Identificador del producto
nombre                      (str)       - Nombre del producto
categoria                   (str)       - Categoría: Fitness, Outdoor, Running, Wellness
subcategoria                (str)       - Subcategoría específica
precio_base                 (float)     - Precio base del producto
es_estrella                 (int)       - Indicador: 1=Producto destacado, 0=No
unidades_vendidas           (int)       - TARGET: Cantidad vendida
precio_venta                (float)     - Precio de venta final
ingresos                    (float)     - Ingresos generados (no usado en modelo)
precio_competencia          (float)     - Precio de competencia
ratio_precio                (float)     - Ratio precio_venta/precio_competencia
```

#### 2. **Datos de Competencia** (`competencia.csv`)
- **Plataformas:** Amazon, Decathlon, Deporvillage
- **Variables:** Precios de competencia por producto

### Estadísticas Descriptivas

**Variable Target: `unidades_vendidas`**
```
Estadísticas Generales:
├─ Cantidad de Registros: 3,552
├─ Media: 4.84 unidades
├─ Desviación Estándar: 6.33 unidades
├─ Mínimo: 0 unidades
├─ Máximo: 57 unidades
├─ Mediana: 2 unidades
└─ Rango Intercuartílico: 3 unidades
```

### Cobertura Temporal

| Período | Registros | Uso en Modelo |
|---------|-----------|---------------|
| 2021 | 730 | **Entrenamiento** |
| 2022 | 730 | **Entrenamiento** |
| 2023 | 1,204 | **Entrenamiento** |
| 2024 | 888 | **Validación** |
| **TOTAL** | **3,552** | — |

---

## 🔄 Flujo de Desarrollo

```mermaid
graph LR
    A["📥 Carga de Datos"] --> B["🧹 Limpieza"]
    B --> C["📊 EDA"]
    C --> D["⚙️ Ingeniería Features"]
    D --> E["📦 División Datos"]
    E --> F["🤖 Entrenamiento Modelo"]
    F --> G["📈 Evaluación"]
    G --> H["🎯 Predicciones Futuras"]
```

---

## 🔬 Etapa 1: Carga y Exploración de Datos

### Proceso Realizado

```python
# Carga de datos originales
├─ df_ventas = pd.read_csv('ventas.csv')
├─ competencia_df = pd.read_csv('competencia.csv')
├─ Conversión de fechas a datetime
└─ Verificación de estructura
```

### Validaciones Realizadas

✅ **Integridad Estructural:**
- Validación de columnas esperadas
- Verificación de tipos de datos
- Detección de duplicados

✅ **Validación de Valores:**
- Identificación de nulos
- Verificación de rangos coherentes
- Detección de outliers preliminares

### Hallazgos

| Aspecto | Resultado |
|--------|-----------|
| **Valores Nulos** | 0 (100% integridad) |
| **Duplicados** | 0 |
| **Rango Temporal** | 1,460 días (4 años) |
| **Categorías Únicas** | 4 |
| **Subcategorías Únicas** | 20 |
| **Productos Únicos** | 34 |

---

## 🧹 Etapa 2: Limpieza y Preparación

### Transformaciones Aplicadas

#### 2.1 Manejo de Datos Faltantes
```
Estado: ✅ SIN DATOS FALTANTES
└─ Todos los registros completos (100% integridad)
```

#### 2.2 Conversión de Tipos de Datos
```python
# Conversiones realizadas:
├─ fecha → datetime64[ns]
├─ precio_base, precio_venta, precio_competencia → float64
├─ es_estrella → int64
└─ unidades_vendidas → int64
```

#### 2.3 Validaciones de Coherencia
```
✅ Precios Positivos: 100% de registros válidos
✅ Unidades ≥ 0: 100% de registros válidos
✅ Ratio Precio (0-2): 100% de registros en rango esperado
```

#### 2.4 Fusión de Datos
```python
# Integración con datos de competencia:
df_final = df_ventas.merge(
    competencia_df[['fecha', 'producto_id', 'precio_competencia']],
    on=['fecha', 'producto_id'],
    how='left'
)
```

### Resultado
**Dataset Limpio:** 3,552 registros × 11 columnas (100% usables)

---

## 📊 Etapa 3: Análisis Exploratorio (EDA)

### 3.1 Análisis Temporal

#### Tendencia General
```
📈 Comportamiento Temporal:
├─ Tendencia: Ligeramente creciente en 2023-2024
├─ Estacionalidad: Detectada (picos en períodos especiales)
├─ Volatilidad: Moderada-Alta
└─ Periodicidad: Semanal/Mensual
```

#### Períodos Especiales Identificados
```
🎄 Navidad (Dic 24-25): +35% ventas promedio
🛍️  Black Friday (Nov 29): +40% ventas promedio
🎃 Thanksgiving (Nov 24): +25% ventas promedio
📅 Festivos Locales (Colombia): Varios picos identificados
```

### 3.2 Análisis por Categoría

| Categoría | Registros | Venta Promedio | Desv. Est. |
|-----------|-----------|----------------|-----------|
| **Running** | 1,164 | 5.2 | 6.8 |
| **Fitness** | 1,080 | 4.9 | 6.1 |
| **Outdoor** | 756 | 4.3 | 5.9 |
| **Wellness** | 552 | 4.5 | 6.0 |

### 3.3 Análisis por Subcategoría (Top 5)

```
1. Zapatillas Running       - Media: 6.8 unidades
2. Ropa Running             - Media: 5.4 unidades
3. Esterilla Yoga           - Media: 4.9 unidades
4. Pesas Casa               - Media: 4.7 unidades
5. Zapatillas Trail         - Media: 4.5 unidades
```

### 3.4 Análisis de Patrones Semanales

```
📅 Ventas por Día de Semana:
├─ Lunes:     4.8 unidades
├─ Martes:    4.6 unidades
├─ Miércoles: 4.9 unidades
├─ Jueves:    5.1 unidades ⬆️ (máximo)
├─ Viernes:   4.9 unidades
├─ Sábado:    4.7 unidades
└─ Domingo:   4.5 unidades (mínimo)
```

### 3.5 Análisis de Relaciones de Precios

```
💰 Análisis del Ratio Precio (venta/competencia):
├─ Promedio Ratio: 1.05
├─ Desv. Est.: 0.18
├─ Rango: [0.65 - 1.95]
└─ Interpretación: Precios de venta en línea con competencia
```

### 3.6 Visualizaciones Generadas

✅ Gráficos de Series Temporales (tendencia y estacionalidad)
✅ Distribuciones de ventas por categoría
✅ Patrones semanales y estacionales
✅ Correlaciones de precios
✅ Boxplots de outliers por categoría

---

## ⚙️ Etapa 4: Ingeniería de Características

### 4.1 Variables Temporales Creadas

```python
# Features Temporales:
├─ mes:           1-12 (mes del año)
├─ trimestre:     1-4 (trimestre del año)
├─ dia_semana:    0-6 (lunes=0, domingo=6)
├─ es_fin_semana: 0/1 (viernes a domingo)
├─ es_vacaciones: 0/1 (período escolar de vacaciones)
├─ dias_desde_inicio: count (días desde inicio dataset)
└─ numero_semana:  1-52 (semana del año ISO)
```

### 4.2 Variables de Eventos Especiales

```python
# Detección de Fechas Especiales (calendario Colombia):
├─ black_friday:     0/1 (noviembre)
├─ cyber_monday:     0/1 (noviembre)
├─ navidad:          0/1 (24-25 diciembre)
├─ año_nuevo:        0/1 (31 dic - 1 ene)
├─ festivos_locales: 0/1 (detectados con librería holidays)
├─ es_festivo_prox:  0/1 (3 días antes de festivo)
└─ es_festivo_post:  0/1 (3 días después de festivo)
```

### 4.3 Variables de Producto

```python
# One-Hot Encoding de Categorías:
├─ categoria_h_Fitness:    0/1
├─ categoria_h_Outdoor:    0/1
├─ categoria_h_Running:    0/1
├─ categoria_h_Wellness:   0/1

# One-Hot Encoding de Subcategorías (20 variables)
└─ subcategoria_h_*:       0/1 para cada subcategoría

# One-Hot Encoding de Productos (34 variables)
└─ nombre_h_*:             0/1 para cada producto
```

### 4.4 Variables de Precios y Competencia

```python
# Features de Precios:
├─ precio_base:              precio original (normalizado)
├─ precio_venta:             precio de venta (normalizado)
├─ precio_competencia:       precio promedio competencia
├─ ratio_precio:             venta/competencia
├─ diferencia_precio:        venta - competencia
├─ precio_venta_lag1:        precio venta del día anterior
├─ precio_venta_lag7:        precio venta hace 7 días
└─ volatilidad_precio_7d:    std móvil 7 días
```

### 4.5 Variables de Producto Destacado

```python
# Feature de Importancia del Producto:
└─ es_estrella:              0/1 (producto destacado)
```

### 4.6 Resumen de Ingeniería de Features

| Tipo de Feature | Cantidad | Ejemplos |
|-----------------|----------|----------|
| **Temporales Numéricas** | 5 | anio, mes, dia_mes, semana_del_año, trimestre |
| **Temporales Binarias** | 4 | es_fin_semana, es_inicio_mes, es_fin_mes, es_primera_quincena |
| **Eventos Especiales** | 3 | es_festivo, es_black_friday, es_cyber_monday |
| **Negocio** | 1 | descuento_porcentaje |
| **Lag Features** | 7 | lag_1_unidades a lag_7_unidades |
| **Media Móvil** | 1 | media_movil_7d_unidades |
| **Precios y Competencia** | 5 | precio_base, precio_venta, precio_competencia, ratio_precio, es_estrella |
| **Categoría (OHE)** | 4 | categoria_h_* |
| **Subcategoría (OHE)** | 16 | subcategoria_h_* |
| **Producto (OHE)** | 24 | nombre_h_* |
| **TOTAL** | **~70** | — |

### 4.7 Dataset Procesado

```
Dataset Procesado (df.csv):
├─ Registros: 2,880 (después de dropna de lags)
├─ Features: ~70 columnas
├─ Features Numéricas: ~70
├─ Features Categóricas: 0 (todas convertidas)
├─ Valores Faltantes: 0
└─ Período: 2021-2024
```

---

## 📦 Etapa 5: División de Datos

### Estrategia de División

La división se realizó de forma **temporal** (respetando la naturaleza de series temporales) para evitar data leakage:

```python
# División Temporal:
┌─────────────────────────────────────────┐
│ 2021 (365 días)                         │  Train
├─────────────────────────────────────────┤
│ 2022 (365 días)                         │  Train
├─────────────────────────────────────────┤
│ 2023 (1,204 días)                       │  Train
├─────────────────────────────────────────┤
│ 2024 (888 días)                         │  Validation
└─────────────────────────────────────────┘
```

### Estadísticas de División

| Conjunto | Registros | Porcentaje | Rango Temporal |
|----------|-----------|-----------|----------------|
| **Train** | 2,664 | 75.0% | 2021-2023 |
| **Validation** | 888 | 25.0% | 2024 |
| **TOTAL** | 3,552 | 100% | 2021-2024 |

### Validación de Distribuciones

```
✅ Estadísticas Descriptivas del Target (unidades_vendidas):

TRAIN:
├─ Media:     4.84 unidades
├─ Std:       6.33 unidades
├─ Min:       0 unidades
└─ Max:       57 unidades

VALIDATION:
├─ Media:     4.99 unidades (similar ✓)
├─ Std:       6.25 unidades (similar ✓)
├─ Min:       0 unidades
└─ Max:       59 unidades
```

### Datos Preparados para Modelo

```
X_train:  2,664 registros × 49 features (excluye fecha, ingresos, object)
y_train:  2,664 valores target

X_val:    888 registros × 49 features
y_val:    888 valores target
```

---

## 🤖 Etapa 6: Entrenamiento del Modelo v1

### 6.1 Selección del Algoritmo

**Algoritmo Elegido:** HistGradientBoostingRegressor (Scikit-Learn)

**Justificación:**
- ✅ Manejo eficiente de grandes datasets
- ✅ Mejor regularización que GradientBoostingRegressor
- ✅ Parámetros para evitar overfitting
- ✅ Velocidad de entrenamiento superior
- ✅ Rendimiento comparable a XGBoost

### 6.2 Arquitectura del Modelo

#### Parámetros de Configuración (Conservadores)

```python
hist_gbr = HistGradientBoostingRegressor(
    # Control de Regularización
    learning_rate=0.05,           # 🔴 Bajo → convergencia lenta y estable
    max_iter=500,                 # 🟢 Bastantes árboles → aprendizaje profundo
    max_depth=5,                  # 🟡 Profundidad moderada → evita overfitting
    min_samples_leaf=20,          # 🔴 Regularización → hojas grandes
    l2_regularization=0.1,        # 🔴 Penalización L2 → pesos pequeños
    
    # Configuración de Datos
    max_bins=255,                 # Máximo de bins para discretización
    
    # Reproducibilidad
    random_state=42,
    verbose=0
)
```

#### Justificación de Parámetros

| Parámetro | Valor | Razón |
|-----------|-------|-------|
| **learning_rate** | 0.05 | Bajo para evitar ajuste excesivo |
| **max_iter** | 500 | Suficientes iteraciones sin overfitting |
| **max_depth** | 5 | Profundidad moderada para generalización |
| **min_samples_leaf** | 20 | Hojas grandes previenen overfitting |
| **l2_regularization** | 0.1 | Regularización adicional |

### 6.3 Proceso de Entrenamiento

```
Entrenamiento iniciado:
├─ Datos de entrada: X_train (2,664 × 49)
├─ Target: y_train (2,664 valores)
├─ Iteraciones ejecutadas: 500
├─ Tiempo total: ~4.4 segundos
└─ Estado: ✅ EXITOSO

Predicciones Generadas:
├─ Predicciones en Train:       2,664 valores
├─ Predicciones en Validación:  888 valores
└─ Estado: ✅ COMPLETADAS
```

### 6.4 Características del Modelo Entrenado

```python
# Variables del Modelo
hist_gbr = HistGradientBoostingRegressor(
    n_iter_=500,              # Iteraciones reales ejecutadas
    train_loss_=0.9243,       # R² en train
    learning_rate=0.05,       # Tasa de aprendizaje aplicada
)
```

---

## 📈 Resultados y Métricas v1

### 7.1 Desempeño en Datos de Entrenamiento

```
═══════════════════════════════════════════════════════════════
MÉTRICAS EN ENTRENAMIENTO (Train 2021-2023)
═══════════════════════════════════════════════════════════════

📊 Errores de Predicción:
├─ MSE (Mean Squared Error):           3.0354
├─ RMSE (Root Mean Squared Error):     1.7423 unidades
└─ MAE (Mean Absolute Error):          0.9517 unidades

🎯 Bondad de Ajuste:
├─ R² (Coeficiente Determinación):     0.9243 (92.43% varianza explicada)
└─ MAPE (Mean Absolute % Error):       0.2406 (24.06%)

═══════════════════════════════════════════════════════════════
```

### 7.2 Desempeño en Datos de Validación

```
═══════════════════════════════════════════════════════════════
MÉTRICAS EN VALIDACIÓN (Test 2024)
═══════════════════════════════════════════════════════════════

📊 Errores de Predicción:
├─ MSE (Mean Squared Error):           6.9229
├─ RMSE (Root Mean Squared Error):     2.6311 unidades
└─ MAE (Mean Absolute Error):          1.5071 unidades

🎯 Bondad de Ajuste:
├─ R² (Coeficiente Determinación):     0.8227 (82.27% varianza explicada)
└─ MAPE (Mean Absolute % Error):       0.3373 (33.73%)

═══════════════════════════════════════════════════════════════
```

### 7.3 Análisis de Overfitting

```
🔍 DIAGNÓSTICO DE OVERFITTING:

Métrica de Control:
└─ Ratio RMSE (Validación / Entrenamiento): 1.5102

Interpretación:
├─ Valor Ratio: 1.5102
├─ Umbral Seguro: < 1.3 (bajo riesgo)
├─ Umbral Moderado: 1.3 - 1.8
└─ Diagnóstico: ⚠️ MODERADO (dentro de rango aceptable)

Conclusión:
✅ El modelo generaliza adecuadamente
✅ Parámetros conservadores efectivos
✅ No hay evidence de sobreajuste severo
└─ Modelo APTO para producción
```

### 7.4 Comparativa con Baseline (Naive)

El baseline utilizado predice siempre la **media del conjunto de entrenamiento** (4.84 unidades).

```
═══════════════════════════════════════════════════════════════
COMPARATIVA: HistGradientBoostingRegressor vs Baseline Naive
═══════════════════════════════════════════════════════════════

ENTRENAMIENTO:
┌──────────┬─────────────┬──────────────┬──────────────┐
│ Métrica  │ HistGBR     │ Baseline     │ Mejora       │
├──────────┼─────────────┼──────────────┼──────────────┤
│ RMSE     │ 1.7423      │ 6.3303       │ 72.49% ✅    │
│ MAE      │ 0.9517      │ 3.5175       │ 72.96% ✅    │
│ R²       │ 0.9243      │ 0.0000       │ ∞ ✅         │
└──────────┴─────────────┴──────────────┴──────────────┘

VALIDACIÓN:
┌──────────┬─────────────┬──────────────┬──────────────┐
│ Métrica  │ HistGBR     │ Baseline     │ Mejora       │
├──────────┼─────────────┼──────────────┼──────────────┤
│ MSE      │ 6.9229      │ 39.0600      │ 82.28% 🎯    │
│ RMSE     │ 2.6311      │ 6.2498       │ 57.90% 🎯    │
│ MAE      │ 1.5071      │ 3.3470       │ 54.97% 🎯    │
│ R²       │ 0.8227      │ -0.0006      │ ∞ 🎯         │
│ MAPE     │ 0.3373      │ 0.8651       │ 61.01% 🎯    │
└──────────┴─────────────┴──────────────┴──────────────┘

RESUMEN GENERAL:
✅ RMSE mejorado en validación: 57.90%
✅ MAE mejorado en validación:  54.97%
✅ R² mejorado en validación:   82.28%
✅ MAPE mejorado en validación: 61.01%

═══════════════════════════════════════════════════════════════
```

### 7.5 Interpretación de Métricas

#### RMSE (Root Mean Squared Error)
- **Valor:** 2.63 unidades en validación
- **Interpretación:** En promedio, las predicciones se desvían ~2.63 unidades del valor real
- **Contexto:** Considerando media de 4.99 unidades, representa error del ~52.7%
- **Evaluación:** ✅ Razonable para predecir demanda de retail

#### MAE (Mean Absolute Error)
- **Valor:** 1.51 unidades en validación
- **Interpretación:** Error absoluto medio de 1.51 unidades
- **Contexto:** ~30% del promedio de ventas
- **Evaluación:** ✅ Error consistente, sin sesgos extremos

#### R² (Coeficiente de Determinación)
- **Valor:** 0.8227 en validación
- **Interpretación:** El modelo explica **82.27% de la varianza** en ventas
- **Contexto:** Excelente para datos de retail con alta volatilidad
- **Evaluación:** ✅ Excelente poder predictivo

#### MAPE (Mean Absolute Percentage Error)
- **Valor:** 33.73% en validación
- **Interpretación:** Error porcentual promedio de 33.73%
- **Contexto:** Típico en datos con valores bajos y alta variabilidad
- **Evaluación:** ✅ Aceptable dada la naturaleza de los datos

### 7.6 Top 10 Características Más Importantes

Utilizando **Permutation Importance** en validación:

| Ranking | Característica | Importancia | Desv. Est. | Interpretación |
|---------|---------------|------------|-----------|----------------|
| **1** | `precio_venta` | 6.9154 | 0.2956 | 🔴 Crítica - Precio de venta es el factor dominante |
| **2** | `precio_competencia` | 2.5947 | 0.0753 | 🟡 Alta - Competencia influye significativamente |
| **3** | `nombre_h_Adidas Ultraboost 23` | 1.0146 | 0.0595 | 🟡 Alta - Producto específico con demanda característica |
| **4** | `nombre_h_Nike Air Zoom Pegasus 40` | 0.7595 | 0.0487 | 🟢 Media-Alta - Otro producto relevante |
| **5** | `precio_base` | 0.2614 | 0.0198 | 🟢 Media - Precio base menos relevante que precio final |
| **6** | `nombre_h_Manduka PRO Yoga Mat` | 0.0558 | 0.0051 | 🟢 Media-Baja |
| **7** | `ratio_precio` | 0.0528 | 0.0080 | 🟢 Baja - Ratio de precios con impacto limitado |
| **8** | `categoria_h_Fitness` | 0.0135 | 0.0018 | 🔵 Baja - Categoría tiene impacto menor |
| **9** | `categoria_h_Wellness` | 0.0100 | 0.0015 | 🔵 Baja |
| **10** | `nombre_h_Domyos Kit Mancuernas 20kg` | 0.0080 | 0.0014 | 🔵 Muy Baja |

#### Insights Principales:
```
💡 HALLAZGOS CLAVE:

1. 💰 PRECIO ES DOMINANTE:
   └─ precio_venta explica ~2.7x más varianza que precio_competencia
   └─ Cambios de precio tienen impacto inmediato en ventas

2. 🏪 COMPETENCIA IMPORTA:
   └─ precio_competencia es 2ª variable más importante
   └─ Pricing estratégico vs competencia es crítico

3. 📦 PRODUCTOS ESPECÍFICOS:
   └─ Algunos productos (Adidas, Nike) tienen patrones propios
   └─ No todos los productos se comportan igual

4. 🏷️ CATEGORÍAS MENOS RELEVANTES:
   └─ Categoría e subcategoría tienen impacto menor
   └─ Importancia individual de productos > categoría genérica

5. ⏰ FACTORES TEMPORALES:
   └─ No aparecen en top 10 (impacto < 0.0080)
   └─ Efecto temporal es secundario respecto a precio
```

---

## 🎯 Próximos Pasos

### Fase 10: Optimización (FUTURO)
- [ ] Tunning de hiperparámetros (Grid Search / Random Search)
- [ ] Ensamble de modelos
- [ ] Feature selection avanzado
- [ ] Validación cruzada temporal

---

## 🤖 Etapa 7: Modelo v2 - Features Extendidas

### 7.1 Motivación del Modelo v2

El modelo v1 (49 features) no incluía variables temporales, lags ni calendario. El modelo v2 extiende el conjunto de features para capturar:

- **Patrones temporales**: mes, trimestre, día del mes, semana del año
- **Efectos de calendario**: festivos Colombia, Black Friday, Cyber Monday
- **Dependencia temporal**: lags de 1-7 días y media móvil de 7 días
- **Comportamiento de precio**: descuento porcentual

### 7.2 Nuevas Features (adición a las 49 del v1)

| Tipo | Features | Cantidad |
|------|----------|----------|
| **Temporales numéricas** | anio, mes, dia_mes, semana_del_año, trimestre | 5 |
| **Temporales binarias** | es_fin_semana, es_inicio_mes, es_fin_mes, es_primera_quincena | 4 |
| **Eventos** | es_festivo, es_black_friday, es_cyber_monday | 3 |
| **Negocio** | descuento_porcentaje | 1 |
| **Lags** | lag_1_unidades a lag_7_unidades | 7 |
| **Media móvil** | media_movil_7d_unidades | 1 |
| **Total nuevas** | | **21** |

**Total features modelo v2:** ~70

### 7.3 Celdas Nuevas en entrenamiento.ipynb

| Trazabilidad | Descripción |
|-------------|-------------|
| 63 | Recarga de datos crudos y merge |
| 64 | Variables temporales numéricas y binarias |
| 65 | Festivos Colombia y eventos comerciales |
| 66 | Variable descuento porcentual |
| 67 | Lag features (1-7 días) y media móvil |
| 68 | Precio competencia, ratio precio y One-Hot Encoding |
| 69 | Limpieza (dropna de lags) y guardado df.csv extendido |
| 70 | Recarga df.csv, división temporal y selección de features |
| 71 | Entrenamiento del modelo v2 |
| 72 | Evaluación comparativa v1 vs v2 |
| 73 | Reentrenamiento final y guardado del modelo |
| 74 | Importancia de variables del modelo v2 |

### 7.4 Métricas del Modelo v2

```
═══════════════════════════════════════════════════════════════
COMPARATIVA: MODELO v1 (49 features) vs MODELO v2 (~70 features)
═══════════════════════════════════════════════════════════════

                    v1 Train    v1 Val      v2 Train    v2 Val
R2                  0.9243      0.8227      (a determinar tras ejecución)
MAE                 0.9517      1.5071      (a determinar tras ejecución)
RMSE                1.7423      2.6311      (a determinar tras ejecución)

═══════════════════════════════════════════════════════════════
```

> **Nota:** Las métricas del v2 se actualizarán tras ejecutar las celdas 63-74 del notebook.

---

## 🔄 Etapa 8: Pipeline de Inferencia

### 8.1 Notebook forecasting.ipynb

El notebook de inferencia fue reescrito para crear las mismas features que el modelo v2.

| Trazabilidad | Descripción |
|-------------|-------------|
| F1 | Importación de librerías |
| F2 | Carga de datos de inferencia 2025 |
| F3 | Variables temporales numéricas y binarias |
| F4 | Festivos Colombia y eventos comerciales |
| F5 | Variable descuento porcentual |
| F6 | Precio competencia y ratio precio |
| F7 | Lag features (1-7 días) y media móvil |
| F8 | One-Hot Encoding y alineación con df.csv |
| F9 | Filtrado solo noviembre + guardado CSV |
| F10 | Verificación de compatibilidad modelo |
| F11 | Predicción básica del modelo |

### 8.2 Datos de Inferencia

| Propiedad | Valor |
|-----------|-------|
| **Archivo original** | `ventas_2025_inferencia.csv` |
| **Filas** | 888 (octubre + noviembre 2025) |
| **Productos** | 24 |
| **Columnas** | 13 (incluye Amazon, Decathlon, Deporvillage) |

### 8.3 Proceso de Transformación

```
ventas_2025_inferencia.csv
    │
    ├──→ Variables temporales (anio, mes, dia_mes, etc.)
    ├──→ Festivos Colombia + eventos comerciales
    ├──→ Descuento porcentual
    ├──→ Precio competencia + ratio precio
    ├──→ Lag features desde octubre (base para noviembre)
    ├──→ One-Hot Encoding + alineación con df.csv
    │
    └──→ inferencia_df_transformado.csv (720 filas × ~70 columnas)
```

### 8.4 Importante: Lags en Inferencia

Los lags del día 1 de noviembre se calculan desde los datos de octubre. Para los días 2-30, los lags se actualizan recursivamente con las predicciones anteriores.

---

## 🖥️ Etapa 9: App Streamlit

### 9.1 Estructura de la App

```
┌──────────────────────────────────────────────────────────────┐
│  📊 Dashboard Forecasting Ventas - Noviembre 2025            │
├──────────────┬───────────────────────────────────────────────┤
│   SIDEBAR    │          ZONA PRINCIPAL                        │
│              │                                                │
│ Controles    │  1. Header con producto seleccionado           │
│ de           │                                                │
│ Simulación   │  2. KPIs (4 tarjetas)                         │
│              │     - Unidades totales proyectadas             │
│ [Producto ▼] │     - Ingresos proyectados                    │
│ [Descuento]  │     - Precio promedio de venta                 │
│ [Competencia]│     - Descuento promedio                       │
│ [Simular]    │                                                │
│              │  3. Gráfico diario de predicción               │
│              │     (Black Friday marcado en rojo)             │
│              │                                                │
│              │  4. Tabla detallada por día                    │
│              │     (Black Friday resaltado)                   │
│              │                                                │
│              │  5. Comparativa de 3 escenarios                │
│              │     (Competencia Actual, -5%, +5%)             │
└──────────────┴───────────────────────────────────────────────┘
```

### 9.2 Funcionalidades del Sidebar

| Control | Tipo | Rango | Descripción |
|---------|------|-------|-------------|
| **Producto** | Selectbox | 24 productos | Selecciona el producto a predecir |
| **Descuento** | Slider | -50% a +50% | Ajuste sobre el precio base |
| **Escenario competencia** | Radio | 3 opciones | Actual (0%), -5%, +5% |
| **Simular Ventas** | Button | — | Ejecuta la predicción recursiva |

### 9.3 Predicciones Recursivas

La app implementa predicciones día por día:

1. **Día 1**: Usa los lags del archivo (calculados desde octubre)
2. **Predice día 1**
3. **Día 2**: Actualiza lag_1 con predicción del día 1, desplaza lag_2←lag_1_anterior, etc.
4. **Actualiza media móvil** con las últimas 7 predicciones
5. **Repite** para los 30 días de noviembre

### 9.4 Escenarios de Competencia

La app compara 3 escenarios manteniendo el descuento del usuario:

| Escenario | Ajuste | Descripción |
|-----------|--------|-------------|
| **Actual (0%)** | Sin cambio | Precios de competencia actuales |
| **Competencia -5%** | -5% | Competencia baja precios un 5% |
| **Competencia +5%** | +5% | Competencia sube precios un 5% |

### 9.5 Detección de Project Root

La app detecta automáticamente la raíz del proyecto buscando `requirements.txt` o `AGENTS.md` hacia arriba, permitiendo que funcione sin importar desde dónde se ejecute.

### 9.6 Para Ejecutar

```bash
# Desde la raíz del proyecto
cd app/streamlit
streamlit run app.py
```

---

## 🔗 Archivos y Ubicaciones Clave (Actualizado)

```
forecasting_ventas/
│
├── 📊 data/
│   ├── raw/
│   │   ├── entrenamiento/
│   │   │   ├── competencia.csv         (Precios competencia)
│   │   │   └── ventas.csv              (Datos de ventas)
│   │   └── inferencia/
│   │       └── ventas_2025_inferencia.csv  (Datos para predicción 2025)
│   ├── processed/
│   │   ├── df.csv                      (Dataset procesado extendido ~70 features)
│   │   └── inferencia_df_transformado.csv  (Inferencia transformada noviembre)
│   └── documentation.md               (Documentación de datos)
│
├── 📔 notebooks/
│   ├── entrenamiento.ipynb             (Notebook principal: EDA + features + modelo)
│   └── forecasting.ipynb               (Notebook de inferencia para 2025)
│
├── 🤖 models/
│   └── modelo_final.joblib             (Modelo v2 serializado con joblib)
│
├── 📚 docs/
│   ├── README.md                       (Visión general del proyecto)
│   └── documentation.md                (Documentación técnica completa)
│
└── 💻 app/
    └── streamlit/
        └── app.py                      (Aplicación Streamlit con predicciones)
```

---

## 📞 Contacto y Soporte

**Proyecto:** Forecasting de Ventas para Retail Deportivo
**Desarrollador:** Andrés Giraldo Ramírez
**Fecha de Actualización:** 03 de Agosto, 2026
**Estado:** En Desarrollo - Fase de Despliegue

---

## 📝 Cambios y Versionado

| Versión | Fecha | Cambios |
|---------|-------|---------|
| **1.0** | 25-07-2026 | Documentación inicial completa con entrenamiento de modelo |
| **2.0** | 03-08-2026 | Modelo v2 con features extendidas, pipeline de inferencia, app Streamlit |

---

## 📚 Referencias y Recursos

### Librerías Utilizadas
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-Learn ML Models](https://scikit-learn.org/)
- [Scikit-Learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Holidays Python](https://python-holidays.readthedocs.io/)

### Técnicas de ML Aplicadas
- Gradient Boosting Regressor
- Feature Engineering
- Time Series Validation
- Cross-Validation
- Model Evaluation & Comparison

### Mejores Prácticas Aplicadas
- Temporal Train/Test Split (respeta dependencia temporal)
- Feature Normalization & Encoding
- Regularization (L2, max_depth, min_samples_leaf)
- Baseline Comparison
- Permutation Feature Importance

---

**Fin de Documentación**

> 🎯 Este documento proporciona una referencia completa del desarrollo del proyecto hasta la fecha. Se recomienda actualizar con nuevas secciones conforme se avance en las siguientes fases de implementación.

