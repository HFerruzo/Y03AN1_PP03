# 1. IMPORTAR LIBRERÍAS
import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Gestión de Compras",
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

# Creación del Menú
menu = st.sidebar.selectbox(
    "Menú Principal",
    ["Datos", "Productos", "Destinos", "Modelamiento Predictivo"]
)

# 2. CARGAR LOS DATOS
df = pd.read_csv("ovas_clientes_sintetico_full.csv")
#print("Filas y columnas:", df.shape)
#df.head()
df_datos = pd.DataFrame(df)
st.dataframe(df_datos, use_container_width=True)