# Documentación de Datos - Forecasting de Ventas

## Estructura del Directorio

```
data/
├── raw/
│   ├── entrenamiento/
│   │   ├── ventas.csv              # Datos de entrenamiento originales
│   │   └── competencia.csv         # Precios de competidores
│   └── inferencia/
│       └── ventas_2025_inferencia.csv  # Datos para predicción 2025
├── processed/
│   ├── df.csv                      # Dataset procesado (entrenamiento)
│   └── inferencia_df_transformado.csv  # Dataset transformado (inferencia)
```

---

## 1. Datos Raw - Entrenamiento

### 1.1 `ventas.csv`

| Propiedad | Valor |
|-----------|-------|
| **Filas** | 3,552 |
| **Columnas** | 10 |
| **Período** | 2021-10-25 a 2024-10-24 |
| **Productos** | 23 |
| **Frecuencia** | Diaria |

#### Columnas

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `fecha` | date | Fecha del registro (YYYY-MM-DD) |
| `producto_id` | str | Identificador del producto (PROD_001 a PROD_023) |
| `nombre` | str | Nombre descriptivo del producto |
| `categoria` | str | Categoría: Running, Fitness, Outdoor, Wellness |
| `subcategoria` | str | Subcategoría (16 valores posibles) |
| `precio_base` | int | Precio base del producto |
| `es_estrella` | bool | True si es producto estrella (5 productos) |
| `unidades_vendidas` | float | Unidades vendidas en el día |
| `precio_venta` | float | Precio de venta final |
| `ingresos` | float | Ingresos totales (precio_venta × unidades_vendidas) |

#### Productos Estrella
- PROD_001: Nike Air Zoom Pegasus 40
- PROD_002: Adidas Ultraboost 23
- PROD_009: Manduka PRO Yoga Mat
- PROD_013: The North Face Borealis
- PROD_016: Domyos Kit Mancuernas 20kg

---

### 1.2 `competencia.csv`

| Propiedad | Valor |
|-----------|-------|
| **Filas** | 3,552 |
| **Columnas** | 5 |
| **Período** | 2021-10-25 a 2024-10-24 |

#### Columnas

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `fecha` | date | Fecha del registro |
| `producto_id` | str | Identificador del producto |
| `Amazon` | float | Precio en Amazon |
| `Decathlon` | float | Precio en Decathlon |
| `Deporvillage` | float | Precio en Deporvillage |

---

## 2. Datos Raw - Inferencia

### 2.1 `ventas_2025_inferencia.csv`

| Propiedad | Valor |
|-----------|-------|
| **Filas** | 888 |
| **Columnas** | 13 |
| **Período** | 2025-10-25 a 2025-11-30 |
| **Productos** | 24 |

#### Columnas

Combina columnas de ventas y competencia en un solo archivo:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `fecha` | date | Fecha del registro |
| `producto_id` | str | Identificador del producto |
| `nombre` | str | Nombre del producto |
| `categoria` | str | Categoría |
| `subcategoria` | str | Subcategoría |
| `precio_base` | int | Precio base |
| `es_estrella` | bool | Si es producto estrella |
| `unidades_vendidas` | float | Unidades vendidas (NaN para predicción) |
| `precio_venta` | float | Precio de venta |
| `ingresos` | float | Ingresos |
| `Amazon` | float | Precio en Amazon |
| `Decathlon` | float | Precio en Decathlon |
| `Deporvillage` | float | Precio en Deporvillage |

**Nota:** Los datos de octubre (2025-10-25 a 2025-10-31) se eliminan en el pipeline de transformación.

---

## 3. Datos Procesados

### 3.1 `df.csv` (Entrenamiento)

| Propiedad | Valor |
|-----------|-------|
| **Filas** | 3,552 |
| **Columnas** | 56 |
| **Generado por** | `notebooks/entrenamiento.ipynb` (Trazabilidad 11) |

#### Estructura de Columnas

| Categoría | Cantidad | Patrón | Descripción |
|-----------|----------|--------|-------------|
| Originales | 10 | `fecha`, `producto_id`, etc. | Columnas del CSV original |
| Competencia | 2 | `precio_competencia`, `ratio_precio` | Derivadas de competidores |
| Producto (OHE) | 24 | `nombre_h_*` | One-Hot Encoding de nombre |
| Categoría (OHE) | 4 | `categoria_h_*` | One-Hot Encoding de categoría |
| Subcategoría (OHE) | 16 | `subcategoria_h_*` | One-Hot Encoding de subcategoría |

#### Transformaciones Aplicadas

1. **Merge de datos**: Se unió `ventas.csv` con `competencia.csv` usando (`fecha`, `producto_id`)
2. **precio_competencia**: Promedio de precios de Amazon, Decathlon y Deporvillage
3. **ratio_precio**: `precio_base / precio_competencia`
4. **One-Hot Encoding**: Variables categóricas convertidas a binarias con sufijo `_h`
5. **Eliminación**: Se eliminaron las columnas de competidores individuales (Amazon, Decathlon, Deporvillage)

---

### 3.2 `inferencia_df_transformado.csv` (Inferencia)

| Propiedad | Valor |
|-----------|-------|
| **Filas** | 720 |
| **Columnas** | 56 |
| **Generado por** | `notebooks/forecasting.ipynb` (Trazabilidad F1-F6) |

#### Transformaciones Aplicadas

1. **Carga**: Se cargó `ventas_2025_inferencia.csv` que ya incluye datos de competencia
2. **Competencia**: Se calculó `precio_competencia` (promedio) y `ratio_precio`
3. **Eliminación**: Se eliminaron columnas Amazon, Decathlon, Deporvillage
4. **One-Hot Encoding**: Variables categóricas convertidas a binarias
5. **Alineación**: Se alinearon las 56 columnas con `df.csv` (columnas faltantes = 0, columnas extra = eliminadas)
6. **Filtrado**: Se eliminaron registros de octubre (2025-10-25 a 2025-10-31), solo noviembre
7. **Guardado**: Se exportó a `data/processed/inferencia_df_transformado.csv`

---

## 4. Pipeline de Transformación

```
┌─────────────────────────────────────────────────────────────┐
│                    ENTRENAMIENTO                            │
├─────────────────────────────────────────────────────────────┤
│  ventas.csv ─────┐                                         │
│                  ├──→ Merge ─→ Competencia ─→ OHE ─→ df.csv│
│  competencia.csv ┘                                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    INFERENCIA                               │
├─────────────────────────────────────────────────────────────┤
│  ventas_2025_inferencia.csv ─→ Competencia ─→ OHE ─→       │
│  (incluye competencia)             │           │            │
│                                    ↓           ↓            │
│                              Alineación con df.csv          │
│                                    │                        │
│                                    ↓                        │
│                              Filtrado (solo noviembre)      │
│                                    │                        │
│                                    ↓                        │
│                     inferencia_df_transformado.csv          │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Variables del Modelo

### Variables Predictoras (49)

| Tipo | Variables |
|------|-----------|
| **Numéricas** | `precio_base`, `precio_venta`, `ingresos`, `precio_competencia`, `ratio_precio` |
| **Producto (OHE)** | `nombre_h_*` (24 variables) |
| **Categoría (OHE)** | `categoria_h_Fitness`, `categoria_h_Outdoor`, `categoria_h_Running`, `categoria_h_Wellness` |
| **Subcategoría (OHE)** | `subcategoria_h_*` (16 variables) |

### Variable Objetivo

- **`unidades_vendidas`**: Unidades vendidas por producto por día

---

## 6. Notas Importantes

1. **División temporal**: Los datos se dividen en entrenamiento (2021-2023) y validación (2024) para evitar data leakage
2. **Productos estrella**: 5 productos con mayor volumen de ventas, identificados como estratégicos
3. **Competencia**: Los precios de competidores solo están disponibles en datos de entrenamiento y en el archivo de inferencia 2025
4. **Filtrado temporal**: Solo se predicen ventas de noviembre 2025 (se eliminan los 7 días de octubre)
5. **Alineación**: El número de columnas en inferencia debe ser exactamente igual al de entrenamiento (56 columnas)
