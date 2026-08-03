import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# CONFIGURACION DE PAGINA
# ============================================================
st.set_page_config(
    page_title="Forecasting Ventas - Noviembre 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ESTILOS CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .kpi-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border-left: 5px solid #667eea;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: bold;
        color: #667eea;
    }
    .kpi-label {
        font-size: 14px;
        color: #666;
    }
    .scenario-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .bf-highlight {
        background-color: #fff3cd !important;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def get_project_root():
    """Encontrar la raiz del proyecto buscando archivos conocidos."""
    current = os.path.dirname(os.path.abspath(__file__))
    while current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, 'requirements.txt')) or \
           os.path.exists(os.path.join(current, 'AGENTS.md')):
            return current
        current = os.path.dirname(current)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = get_project_root()


@st.cache_resource
def load_model():
    """Cargar el modelo entrenado desde la ruta absoluta del proyecto."""
    model_path = os.path.join(PROJECT_ROOT, 'notebooks', 'models', 'modelo_final.joblib')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


@st.cache_data
def load_inference_data():
    """Cargar el dataset de inferencia transformado desde la ruta absoluta del proyecto."""
    csv_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'inferencia_df_transformado.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


def get_feature_columns(model):
    """Obtener las columnas de features que espera el modelo."""
    return list(model.feature_names_in_)


def prepare_features_for_prediction(row_df, feature_cols, discount_pct, competition_scenario):
    """Preparar las features para una prediccion dada."""
    df = row_df.copy()

    # Aplicar descuento al precio de venta
    df['precio_venta'] = df['precio_base'] * (1 - discount_pct / 100)

    # Ajustar precios de competencia segun escenario
    scenario_multipliers = {
        'Actual (0%)': 1.0,
        'Competencia -5%': 0.95,
        'Competencia +5%': 1.05
    }
    mult = scenario_multipliers.get(competition_scenario, 1.0)

    # Recalcular precio_competencia y ratio_precio
    # Necesitamos las columnas Amazon, Decathlon, Deporvillage originales
    # pero ya fueron eliminadas. Usamos precio_competencia como base.
    if 'precio_competencia' in df.columns:
        df['precio_competencia'] = df['precio_competencia'] * mult
        df['ratio_precio'] = np.where(
            df['precio_competencia'] != 0,
            df['precio_base'] / df['precio_competencia'],
            np.nan
        )

    # Recalcular descuento_porcentaje
    df['descuento_porcentaje'] = ((df['precio_venta'] - df['precio_base']) / df['precio_base'] * 100)

    # Asegurar que todas las features necesarias existen
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    return df[feature_cols]


def recursive_predict(model, base_df, feature_cols, product_df, discount_pct, competition_scenario, days=30):
    """Realizar predicciones recursivas dia por dia para un producto."""
    predictions = []
    daily_data = []

    # Obtener datos del producto ordenados por fecha
    prod_data = product_df.sort_values('fecha').reset_index(drop=True)

    # Para los primeros 7 dias, usar los lags que ya estan en el archivo
    # Para dias siguientes, actualizar lags recursivamente
    lag_values = {}
    for lag in range(1, 8):
        col_name = f'lag_{lag}_unidades'
        if col_name in prod_data.columns:
            lag_values[lag] = prod_data[col_name].values.copy()

    ma7_values = prod_data['media_movil_7d_unidades'].values.copy() if 'media_movil_7d_unidades' in prod_data.columns else np.zeros(len(prod_data))

    for day_idx in range(min(days, len(prod_data))):
        row = prod_data.iloc[day_idx:day_idx+1].copy()

        # Aplicar descuento al precio de venta
        row['precio_venta'] = row['precio_base'] * (1 - discount_pct / 100)

        # Ajustar competencia
        scenario_multipliers = {
            'Actual (0%)': 1.0,
            'Competencia -5%': 0.95,
            'Competencia +5%': 1.05
        }
        mult = scenario_multipliers.get(competition_scenario, 1.0)
        if 'precio_competencia' in row.columns:
            row['precio_competencia'] = row['precio_competencia'] * mult
            row['ratio_precio'] = np.where(
                row['precio_competencia'] != 0,
                row['precio_base'] / row['precio_competencia'],
                np.nan
            )

        # Recalcular descuento_porcentaje
        row['descuento_porcentaje'] = ((row['precio_venta'] - row['precio_base']) / row['precio_base'] * 100)

        # Actualizar lags si no es el primer dia
        if day_idx > 0:
            # Actualizar lag_1 con la prediccion del dia anterior
            row['lag_1_unidades'] = predictions[-1]

            # Desplazar lags: lag_N = valor anterior de lag_(N-1)
            for lag in range(2, 8):
                prev_lag_col = f'lag_{lag-1}_unidades'
                curr_lag_col = f'lag_{lag}_unidades'
                if prev_lag_col in row.columns and curr_lag_col in row.columns:
                    row[curr_lag_col] = row[prev_lag_col]

            # Actualizar media movil con las ultimas 7 predicciones
            recent_preds = predictions[-7:] if len(predictions) >= 7 else predictions
            if len(recent_preds) > 0:
                row['media_movil_7d_unidades'] = np.mean(recent_preds)

        # Asegurar que todas las features existen
        for col in feature_cols:
            if col not in row.columns:
                row[col] = 0

        # Predecir
        X = row[feature_cols]
        pred = model.predict(X)[0]
        pred = max(0, pred)  # No puede ser negativo
        predictions.append(pred)

        # Guardar datos del dia
        dia_info = {
            'fecha': row['fecha'].values[0],
            'dia_mes': row['dia_mes'].values[0] if 'dia_mes' in row.columns else day_idx + 1,
            'precio_venta': row['precio_venta'].values[0],
            'precio_base': row['precio_base'].values[0],
            'precio_competencia': row['precio_competencia'].values[0] if 'precio_competencia' in row.columns else 0,
            'ratio_precio': row['ratio_precio'].values[0] if 'ratio_precio' in row.columns else 0,
            'descuento_porcentaje': row['descuento_porcentaje'].values[0],
            'unidades_predichas': pred,
            'es_black_friday': bool(row['es_black_friday'].values[0]) if 'es_black_friday' in row.columns else False,
            'es_festivo': bool(row['es_festivo'].values[0]) if 'es_festivo' in row.columns else False,
            'es_fin_semana': bool(row['es_fin_semana'].values[0]) if 'es_fin_semana' in row.columns else False,
        }
        daily_data.append(dia_info)

    return predictions, daily_data


# ============================================================
# CARGA DE RECURSOS
# ============================================================
modelo = load_model()
inference_df = load_inference_data()

if modelo is None:
    st.error("No se pudo cargar el modelo. Verifique que el archivo modelo_final.joblib existe.")
    st.stop()

if inference_df is None:
    st.error("No se pudo cargar el dataset de inferencia. Verifique que inferencia_df_transformado.csv existe.")
    st.stop()

feature_cols = get_feature_columns(modelo)

# ============================================================
# SIDEBAR - CONTROLES DE SIMULACION
# ============================================================
with st.sidebar:
    st.markdown("## 🎛️ Controles de Simulación")

    st.markdown("---")

    # Selector de producto
    productos = sorted(inference_df['nombre'].unique().tolist())
    producto_seleccionado = st.selectbox(
        "📦 Seleccionar Producto",
        productos,
        index=0,
        help="Seleccione el producto para el que desea generar predicciones"
    )

    st.markdown("---")

    # Slider de descuento
    descuento = st.slider(
        "💰 Ajuste de Descuento",
        min_value=-50,
        max_value=50,
        value=0,
        step=5,
        format="%d%%",
        help="Ajuste el descuento sobre el precio base (-50% a +50%)"
    )

    st.markdown("---")

    # Selector de escenario de competencia
    st.markdown("🏢 Escenario de Competencia")
    escenario_competencia = st.radio(
        "Precio competencia:",
        ["Actual (0%)", "Competencia -5%", "Competencia +5%"],
        index=0,
        help="Seleccione como cambian los precios de la competencia"
    )

    st.markdown("---")

    # Boton de simulacion
    simular = st.button(
        "🚀 Simular Ventas",
        type="primary",
        use_container_width=True,
        help="Ejecutar la prediccion recursiva para el producto seleccionado"
    )


# ============================================================
# ZONA PRINCIPAL - DASHBOARD
# ============================================================

# Header
st.markdown(f"""
<div class="main-header">
    <h1>📊 Dashboard Forecasting Ventas - Noviembre 2025</h1>
    <p>Producto: <strong>{producto_seleccionado}</strong> | Descuento: <strong>{descuento}%</strong> | Competencia: <strong>{escenario_competencia}</strong></p>
</div>
""", unsafe_allow_html=True)

if simular:
    with st.spinner("Generando predicciones recursivas..."):
        # Obtener datos del producto seleccionado
        product_df = inference_df[inference_df['nombre'] == producto_seleccionado].copy()

        if len(product_df) == 0:
            st.error(f"No se encontraron datos para el producto: {producto_seleccionado}")
            st.stop()

        # Obtener info basica del producto
        precio_base = product_df['precio_base'].values[0]
        es_estrella = product_df['es_estrella'].values[0]
        producto_id = product_df['producto_id'].values[0]

        # Realizar predicciones recursivas
        predictions, daily_data = recursive_predict(
            modelo, inference_df, feature_cols, product_df,
            descuento, escenario_competencia, days=30
        )

        # Crear DataFrame de resultados
        results_df = pd.DataFrame(daily_data)
        results_df['dia_semana'] = pd.to_datetime(results_df['fecha']).dt.day_name()

        # ============================================================
        # KPIs
        # ============================================================
        total_unidades = sum(predictions)
        precio_venta_final = precio_base * (1 - descuento / 100)
        ingresos_total = total_unidades * precio_venta_final
        descuento_promedio = descuento

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{total_unidades:.0f}</div>
                <div class="kpi-label">Unidades Totales Proyectadas</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">€{ingresos_total:,.2f}</div>
                <div class="kpi-label">Ingresos Proyectados</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">€{precio_venta_final:.2f}</div>
                <div class="kpi-label">Precio Promedio de Venta</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{descuento_promedio}%</div>
                <div class="kpi-label">Descuento Aplicado</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ============================================================
        # GRAFICO DE PREDICCION DIARIA
        # ============================================================
        st.markdown("### 📈 Predicción Diaria de Ventas")

        fig = go.Figure()

        # Linea de prediccion
        fig.add_trace(go.Scatter(
            x=results_df['fecha'],
            y=results_df['unidades_predichas'],
            mode='lines+markers',
            name='Unidades Predichas',
            line=dict(color='#667eea', width=3),
            marker=dict(size=6, color='#667eea')
        ))

        # Marcar Black Friday
        bf_row = results_df[results_df['es_black_friday'] == True]
        if len(bf_row) > 0:
            bf_fecha = bf_row['fecha'].values[0]
            bf_unidades = bf_row['unidades_predichas'].values[0]

            # Linea vertical en Black Friday
            fig.add_vline(
                x=bf_fecha,
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text="Black Friday",
                annotation_position="top"
            )

            # Punto resaltado
            fig.add_trace(go.Scatter(
                x=[bf_fecha],
                y=[bf_unidades],
                mode='markers',
                name='Black Friday',
                marker=dict(size=15, color='red', symbol='star'),
                showlegend=True
            ))

        # Marcar festivos
        festivos = results_df[results_df['es_festivo'] == True]
        if len(festivos) > 0:
            fig.add_trace(go.Scatter(
                x=festivos['fecha'],
                y=festivos['unidades_predichas'],
                mode='markers',
                name='Festivos',
                marker=dict(size=10, color='orange', symbol='diamond'),
                showlegend=True
            ))

        fig.update_layout(
            title=dict(
                text=f'Unidades Vendidas Predichas - {producto_seleccionado}',
                font=dict(size=16)
            ),
            xaxis_title='Fecha',
            yaxis_title='Unidades Vendidas',
            template='plotly_white',
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # ============================================================
        # TABLA DETALLADA
        # ============================================================
        st.markdown("### 📋 Tabla Detallada por Dia")

        # Formatear tabla
        display_df = results_df.copy()
        display_df['fecha'] = pd.to_datetime(display_df['fecha']).dt.strftime('%Y-%m-%d')
        display_df['precio_venta'] = display_df['precio_venta'].apply(lambda x: f'€{x:.2f}')
        display_df['precio_competencia'] = display_df['precio_competencia'].apply(lambda x: f'€{x:.2f}')
        display_df['descuento_porcentaje'] = display_df['descuento_porcentaje'].apply(lambda x: f'{x:.1f}%')
        display_df['unidades_predichas'] = display_df['unidades_predichas'].apply(lambda x: f'{x:.0f}')
        display_df['ingresos'] = (results_df['unidades_predichas'] * results_df['precio_venta']).apply(lambda x: f'€{x:,.2f}')

        # Renombrar columnas
        display_df = display_df.rename(columns={
            'fecha': 'Fecha',
            'dia_semana': 'Dia Semana',
            'precio_venta': 'Precio Venta',
            'precio_competencia': 'Precio Competencia',
            'descuento_porcentaje': 'Descuento',
            'unidades_predichas': 'Unidades',
            'ingresos': 'Ingresos',
            'es_black_friday': 'Black Friday',
            'es_festivo': 'Festivo'
        })

        # Seleccionar columnas a mostrar
        cols_show = ['Fecha', 'Dia Semana', 'Precio Venta', 'Precio Competencia',
                     'Descuento', 'Unidades', 'Ingresos', 'Black Friday', 'Festivo']
        display_df = display_df[cols_show]

        # Resaltar Black Friday
        def highlight_bf(row):
            if row.get('Black Friday', False):
                return ['background-color: #fff3cd'] * len(row)
            return [''] * len(row)

        st.dataframe(
            display_df.style.apply(highlight_bf, axis=1),
            use_container_width=True,
            height=400
        )

        st.markdown("---")

        # ============================================================
        # COMPARATIVA DE ESCENARIOS
        # ============================================================
        st.markdown("### 🔄 Comparativa de Escenarios de Competencia")

        escenarios = ['Actual (0%)', 'Competencia -5%', 'Competencia +5%']
        resultados_escenarios = {}

        for esc in escenarios:
            pred_esc, _ = recursive_predict(
                modelo, inference_df, feature_cols, product_df,
                descuento, esc, days=30
            )
            total_esc = sum(pred_esc)
            ingresos_esc = total_esc * precio_venta_final
            resultados_escenarios[esc] = {
                'unidades': total_esc,
                'ingresos': ingresos_esc
            }

        col1, col2, col3 = st.columns(3)

        for idx, esc in enumerate(escenarios):
            with [col1, col2, col3][idx]:
                res = resultados_escenarios[esc]
                color = '#28a745' if esc == 'Actual (0%)' else ('#dc3545' if '-5%' in esc else '#007bff')
                st.markdown(f"""
                <div class="scenario-card">
                    <h4 style="color: {color};">{esc}</h4>
                    <p><strong>Unidades:</strong> {res['unidades']:.0f}</p>
                    <p><strong>Ingresos:</strong> €{res['ingresos']:,.2f}</p>
                </div>
                """, unsafe_allow_html=True)

        # Grafico comparativo
        fig_comp = go.Figure()

        escenario_names = list(resultados_escenarios.keys())
        unidades_vals = [resultados_escenarios[e]['unidades'] for e in escenario_names]
        ingresos_vals = [resultados_escenarios[e]['ingresos'] for e in escenario_names]

        fig_comp.add_trace(go.Bar(
            name='Unidades',
            x=escenario_names,
            y=unidades_vals,
            marker_color=['#28a745', '#dc3545', '#007bff'],
            text=[f'{v:.0f}' for v in unidades_vals],
            textposition='auto'
        ))

        fig_comp.update_layout(
            title='Comparativa de Unidades por Escenario',
            yaxis_title='Unidades Totales',
            template='plotly_white',
            height=300
        )

        st.plotly_chart(fig_comp, use_container_width=True)

else:
    # Mensaje inicial cuando no se ha ejecutado la simulacion
    st.info("👆 Seleccione un producto y configure los controles en el panel lateral, luego presione 'Simular Ventas'.")

    # Mostrar informacion basica
    st.markdown("### 📊 Información del Dataset de Inferencia")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Registros", f"{len(inference_df):,}")
    with col2:
        st.metric("Productos", f"{inference_df['nombre'].nunique()}")
    with col3:
        st.metric("Dias de Noviembre", "30")

    st.markdown("### 📦 Productos Disponibles")
    productos_info = inference_df.groupby('nombre').agg({
        'producto_id': 'first',
        'categoria': 'first',
        'precio_base': 'first',
        'es_estrella': 'first'
    }).reset_index()

    st.dataframe(productos_info, use_container_width=True)
