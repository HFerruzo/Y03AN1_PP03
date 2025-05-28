import streamlit as st
import pandas as pd
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Gestión",
    page_icon="📊",
    layout="wide"
)

# Título central en la parte superior
st.markdown(
    """
    <style>
    .title-style {
        font-size: 36px;
        text-align: center;
        margin-bottom: 30px;
        color: #1f77b4;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="title-style">SISTEMA DE GESTIÓN INTEGRAL</h1>', unsafe_allow_html=True)

# Menú en el costado izquierdo
menu = st.sidebar.selectbox(
    "Menú Principal",
    ["Clientes", "Productos", "Destinos", "Modelamiento Predictivo"]
)

# Contenido según la selección del menú
if menu == "Clientes":
    st.header("📋 Tabla de Clientes")
    
    # Datos de ejemplo
    clientes_data = {
        "ID": [1, 2, 3, 4, 5],
        "Nombre": ["Juan Pérez", "María García", "Carlos López", "Ana Martínez", "Luis Rodríguez"],
        "Email": ["juan@email.com", "maria@email.com", "carlos@email.com", "ana@email.com", "luis@email.com"],
        "Teléfono": ["555-1234", "555-5678", "555-9012", "555-3456", "555-7890"],
        "Última Compra": ["2023-01-15", "2023-02-20", "2023-03-10", "2023-04-05", "2023-05-12"]
    }
    
    df_clientes = pd.DataFrame(clientes_data)
    st.dataframe(df_clientes, use_container_width=True)
    
    # Formulario para agregar nuevo cliente
    with st.expander("Agregar Nuevo Cliente"):
        with st.form("form_cliente"):
            nombre = st.text_input("Nombre Completo")
            email = st.text_input("Email")
            telefono = st.text_input("Teléfono")
            
            submitted = st.form_submit_button("Guardar Cliente")
            if submitted:
                st.success(f"Cliente {nombre} agregado correctamente")

elif menu == "Productos":
    st.header("📦 Tabla de Productos")
    
    productos_data = {
        "ID": [101, 102, 103, 104, 105],
        "Nombre": ["Laptop", "Teléfono", "Tablet", "Monitor", "Teclado"],
        "Categoría": ["Electrónica", "Electrónica", "Electrónica", "Electrónica", "Accesorio"],
        "Precio": [1200, 800, 450, 300, 50],
        "Stock": [15, 30, 20, 25, 50]
    }
    
    df_productos = pd.DataFrame(productos_data)
    st.dataframe(df_productos, use_container_width=True)
    
    # Gráfico de stock
    st.bar_chart(df_productos.set_index("Nombre")["Stock"])

elif menu == "Destinos":
    st.header("✈️ Tabla de Destinos")
    
    destinos_data = {
        "ID": [201, 202, 203, 204, 205],
        "Ciudad": ["Madrid", "Barcelona", "París", "Roma", "Berlín"],
        "País": ["España", "España", "Francia", "Italia", "Alemania"],
        "Costo Envío (€)": [15, 12, 25, 22, 20],
        "Tiempo Entrega (días)": [2, 1, 3, 3, 4]
    }
    
    df_destinos = pd.DataFrame(destinos_data)
    st.dataframe(df_destinos, use_container_width=True)
    
    # Mapa de destinos
    map_data = pd.DataFrame({
        "lat": [40.4168, 41.3851, 48.8566, 41.9028, 52.5200],
        "lon": [-3.7038, 2.1734, 2.3522, 12.4964, 13.4050],
        "name": df_destinos["Ciudad"].tolist()
    })
    
    st.map(map_data, zoom=4)

elif menu == "Modelamiento Predictivo":
    st.header("🔮 Modelamiento Predictivo")
    
    st.write("""
    Esta sección permite realizar predicciones basadas en datos históricos.
    Seleccione los parámetros y ejecute el modelo.
    """)
    
    # Parámetros del modelo
    with st.form("modelo_form"):
        st.subheader("Parámetros del Modelo")
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators = st.slider("Número de estimadores", 10, 200, 100)
            max_depth = st.selectbox("Profundidad máxima", [None, 5, 10, 20])
            
        with col2:
            learning_rate = st.slider("Tasa de aprendizaje", 0.01, 1.0, 0.1)
            random_state = st.number_input("Semilla aleatoria", value=42)
            
        submitted = st.form_submit_button("Ejecutar Modelo")
        
        if submitted:
            # Simulación de entrenamiento del modelo
            with st.spinner("Entrenando modelo..."):
                import time
                time.sleep(2)
                
                # Resultados simulados
                st.success("Modelo entrenado exitosamente!")
                st.metric("Precisión del modelo", "89.3%", "2.1%")
                
                # Gráfico de importancia de características (simulado)
                chart_data = pd.DataFrame({
                    "Característica": ["Edad", "Ingresos", "Historial", "Ubicación", "Dispositivo"],
                    "Importancia": [0.25, 0.35, 0.15, 0.10, 0.15]
                })
                
                st.bar_chart(chart_data.set_index("Característica"))