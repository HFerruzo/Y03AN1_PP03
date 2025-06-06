import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
@st.cache_data
def load_data():
    return pd.read_csv("nuevos_clientes_predicciones.csv")

df = load_data()

st.title("Predicción de Interés de Clientes Nuevos")

# Filtros interactivas
st.sidebar.header("Filtros")
tipo_cliente = st.sidebar.multiselect("Tipo de Cliente", options=df['Tipo_cliente'].unique(), default=df['Tipo_cliente'].unique())
segmento = st.sidebar.multiselect("Segmento", options=df['Segmento'].unique(), default=df['Segmento'].unique())
canal = st.sidebar.multiselect("Canal Principal", options=df['Canal_principal'].unique(), default=df['Canal_principal'].unique())

# Aplicar filtros
filtro_df = df[(df['Tipo_cliente'].isin(tipo_cliente)) &
               (df['Segmento'].isin(segmento)) &
               (df['Canal_principal'].isin(canal))]

st.subheader("Vista Previa de los Datos Filtrados")
st.dataframe(filtro_df.head(10))

# Métricas clave
total = filtro_df.shape[0]
interesados = filtro_df['Predicción_Cliente_Interesado'].sum()
no_interesados = total - interesados

col1, col2 = st.columns(2)
col1.metric("Total Clientes", total)
col2.metric("% Interesados", f"{(interesados / total * 100):.1f}%")

# Gráfico de distribución
st.subheader("Distribución de Predicciones")
fig1, ax1 = plt.subplots()
sns.countplot(data=filtro_df, x='Predicción_Cliente_Interesado', palette='Set2', ax=ax1)
ax1.set_xticklabels(['No Interesado', 'Interesado'])
ax1.set_title("Conteo de Clientes por Predicción")
st.pyplot(fig1)

# Histograma de tasa de respuesta
st.subheader("Distribución de Tasa de Respuesta")
fig2, ax2 = plt.subplots()
sns.histplot(data=filtro_df, x='Tasa_respuesta', hue='Predicción_Cliente_Interesado', kde=True, palette='Set1', ax=ax2)
ax2.set_title("Histograma de Tasa de Respuesta")
st.pyplot(fig2)

st.markdown("---")
st.caption("Dashboard generado con Streamlit")
