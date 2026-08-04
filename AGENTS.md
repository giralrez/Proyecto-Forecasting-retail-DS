# AGENTS.md - Instrucciones para Agentes de Código


## instrucciones generales
-  empieza siempre tu respuesta con el emoji 🤖
-  responde siempre en español
-  para cada instruccion genera un bloque nuevo de codigo y comentalo para llevar una trazabilidad del desarrollo del proyecto.

## Contexto del Proyecto

Proyecto de **Forecasting de Ventas** para retail de artículos deportivos. Predicción de unidades vendidas usando modelos de la familia XGBoost y visualización con Streamlit.

## Comandos Clave

- **Instalar dependencias:** `pip install -r requirements.txt`
- **Ejecutar app:** `streamlit run app/streamlit/app.py`
- **Abrir notebook:** `jupyter notebook notebooks/entrenamiento.ipynb`

## Stack Tecnológico

- Python 3.13
- pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit, plotly, holidays

## Convenciones del Proyecto

### Trazabilidad en Notebooks

Cada celda de código debe incluir un comentario de trazabilidad al inicio:

```python
# Trazabilidad XX - Descripción breve de lo que hace la celda
```

Donde `XX` es un número secuencial creciente.

### Variables Permitidas del DataFrame `df`

Las únicas variables disponibles en el dataframe `df` son:

| Categoría | Variables |
|-----------|-----------|
| **Originales** | `fecha`, `producto_id`, `nombre`, `categoria`, `subcategoria`, `precio_base`, `es_estrella`, `unidades_vendidas`, `precio_venta`, `ingresos` |
| **Competencia** | `precio_competencia`, `ratio_precio` |
| **Producto (OHE)** | `nombre_h_*` (24 variables dummy) |
| **Categoría (OHE)** | `categoria_h_Fitness`, `categoria_h_Outdoor`, `categoria_h_Running`, `categoria_h_Wellness` |
| **Subcategoría (OHE)** | `subcategoria_h_*` (16 variables dummy) |

### Variables del DataFrame `competencia_df`

| Variable | Descripción |
|----------|-------------|
| `fecha` | Fecha |
| `producto_id` | ID del producto |
| `Amazon` | Precio en Amazon |
| `Decathlon` | Precio en Decathlon |
| `Deporvillage` | Precio en Deporvillage |

### Librerías Permitidas

Solo usar las siguientes librerías:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `jupyter`
- `streamlit`
- `holidays`
- `plotly`

**NO** usar: `xgboost`, `lightgbm`, `tensorflow`, `keras`, ni ninguna otra librería no listada.

### Organización de Código

- **`notebooks/`**: Análisis exploratorio, feature engineering y experimentación
- **`app/streamlit/`**: Aplicación web para predicciones
- **`data/`**: Datos raw, processed e inference
- **`models/`**: Modelos entrenados serializados (joblib)
- **`src/`**: Código fuente reutilizable (futuro)

### Reglas de Código

1. No usar variables que no estén en la lista de `df` o `competencia_df`, salvo que se definan en el mismo código
2. No usar librerías fuera del stack permitido
3. Cada celda nueva debe llevar su comentario de trazabilidad
4. Seguir estilo de código existente en el notebook
5. Respaldar la división temporal train/validation (no data leakage)

### Estructura del Dataset Procesado

El archivo `data/processed/df.csv` contiene 56 columnas resultantes del pipeline de feature engineering:
- 10 columnas originales
- 2 columnas derivadas de competencia (`precio_competencia`, `ratio_precio`)
- 24 One-Hot Encodings de producto (`nombre_h_*`)
- 4 One-Hot Encodings de categoría (`categoria_h_*`)
- 16 One-Hot Encodings de subcategoría (`subcategoria_h_*`)

## Archivos Clave

| Archivo | Propósito |
|---------|-----------|
| `notebooks/entrenamiento.ipynb` | Notebook principal con EDA, feature engineering y modelo |
| `app/streamlit/app.py` | Aplicación Streamlit (en desarrollo) |
| `data/raw/entrenamiento/ventas.csv` | Datos de entrenamiento originales |
| `data/raw/entrenamiento/competencia.csv` | Datos de precios de competencia |
| `data/raw/inferencia/ventas_2025_inferencia.csv` | Datos para predicción 2025 |
| `data/processed/df.csv` | Dataset procesado con features |
| `.github/copilot-instructions.md` | Instrucciones detalladas de variables |

## Estado del Proyecto

- [x] Carga y exploración de datos
- [x] Limpieza y validación
- [x] Análisis exploratorio (EDA)
- [x] Ingeniería de características
- [x] División temporal de datos
- [x] Entrenamiento de modelo base (HistGradientBoostingRegressor)
- [x] Evaluación de desempeño (R²=0.82 en validación)
- [ ] Serialización del modelo (joblib)
- [ ] Predicción en datos de inferencia 2025
- [ ] Despliegue con Streamlit
- [ ] Optimización de hiperparámetros

inferencia_df: (['fecha', 'producto_id', 'nombre', 'categoria', 'subcategoria',
       'precio_base', 'es_estrella', 'unidades_vendidas', 'precio_venta',
       'ingresos', 'precio_competencia', 'ratio_precio',
       'nombre_h_Adidas Own The Run Jacket', 'nombre_h_Adidas Ultraboost 23',
       'nombre_h_Asics Gel Nimbus 25', 'nombre_h_Bowflex SelectTech 552',
       'nombre_h_Columbia Silver Ridge',
       'nombre_h_Decathlon Bandas Elásticas Set', 'nombre_h_Domyos BM900',
       'nombre_h_Domyos Kit Mancuernas 20kg',
       'nombre_h_Gaiam Premium Yoga Block', 'nombre_h_Liforme Yoga Pad',
       'nombre_h_Lotuscrafts Yoga Bolster', 'nombre_h_Manduka PRO Yoga Mat',
       'nombre_h_Merrell Moab 2 GTX',
       'nombre_h_New Balance Fresh Foam X 1080v12',
       'nombre_h_Nike Air Zoom Pegasus 40', 'nombre_h_Nike Dri-FIT Miler',
       'nombre_h_Puma Velocity Nitro 2', 'nombre_h_Quechua MH500',
       'nombre_h_Reebok Floatride Energy 5',
       'nombre_h_Reebok Professional Deck',
       'nombre_h_Salomon Speedcross 5 GTX', 'nombre_h_Sveltus Kettlebell 12kg',
       'nombre_h_The North Face Borealis', 'nombre_h_Trek Marlin 7',
       'categoria_h_Fitness', 'categoria_h_Outdoor', 'categoria_h_Running',
       'categoria_h_Wellness', 'subcategoria_h_Banco Gimnasio',
       'subcategoria_h_Bandas Elásticas', 'subcategoria_h_Bicicleta Montaña',
       'subcategoria_h_Bloque Yoga', 'subcategoria_h_Cojín Yoga',
       'subcategoria_h_Esterilla Fitness', 'subcategoria_h_Esterilla Yoga',
       'subcategoria_h_Mancuernas Ajustables',
       'subcategoria_h_Mochila Trekking', 'subcategoria_h_Pesa Rusa',
       'subcategoria_h_Pesas Casa', 'subcategoria_h_Rodillera Yoga',
       'subcategoria_h_Ropa Montaña', 'subcategoria_h_Ropa Running',
       'subcategoria_h_Zapatillas Running', 'subcategoria_h_Zapatillas Trail'],
      dtype='str')