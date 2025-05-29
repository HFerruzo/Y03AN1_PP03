import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración inicial
st.set_page_config(page_title="Análisis de Clientes", layout="wide")
st.title("📊 Análisis de Clientes Interesados en OVAS")

@st.cache_data
def cargar_datos():
    df = pd.read_csv("ovas_clientes_sintetico_full.csv")
    return df

df = cargar_datos()
st.sidebar.title("Navegación")
opcion = st.sidebar.radio("Selecciona una opción", [
    "Vista general",
    "Análisis Exploratorio",
    "Modelo de Clasificación",
    "Importancia de Variables",
    "Predicciones"
])

# Preprocesamiento básico y división
def preprocesamiento(df):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in split.split(df, df['Cliente_interesado']):
        strat_train_set = df.loc[train_idx]
        strat_test_set = df.loc[test_idx]
    df_train = strat_train_set.copy()
    df_test = strat_test_set.copy()

    cols_to_drop = ['ID_cliente', 'Fecha_contacto', 'Ultima_compra', 'Cliente_retenible', 
                    'Probabilidad_retencion', 'Vendedor','Interes_compra','Probabilidad_retencion',
                    'Monto_total_gastado','Satisfaccion']
    df_train = df_train.drop(columns=cols_to_drop)
    df_test = df_test.drop(columns=cols_to_drop)

    for col in ['Cantidad_ovas_compradas', 'Frecuencia_compras', 'Tiempo_respuesta_horas']:
        df_train[col + '_bin'] = pd.qcut(df_train[col], q=4, labels=False, duplicates='drop')
        df_test[col + '_bin'] = pd.qcut(df_test[col], q=4, labels=False, duplicates='drop')
        df_train = df_train.drop(columns=[col])
        df_test = df_test.drop(columns=[col])

    cat_cols = df_train.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in cat_cols:
        df_train[col] = le.fit_transform(df_train[col])
        df_test[col] = le.transform(df_test[col])

    X_train = df_train.drop('Cliente_interesado', axis=1)
    y_train = df_train['Cliente_interesado']
    X_test = df_test.drop('Cliente_interesado', axis=1)
    y_test = df_test['Cliente_interesado']
    return X_train, X_test, y_train, y_test, df_train, df_test

X_train, X_test, y_train, y_test, df_train, df_test = preprocesamiento(df)

if opcion == "Vista general":
    st.subheader("Información del Dataset")
    st.write(df.head())
    st.write("Distribución de la variable objetivo:")
    fig = px.histogram(df, x='Cliente_interesado', color='Cliente_interesado', barmode='group')
    st.plotly_chart(fig, use_container_width=True)

elif opcion == "Análisis Exploratorio":
    st.subheader("Distribución de Variables Numéricas")
    numeric_cols = df.select_dtypes(include=np.number).columns
    col1, col2 = st.columns(2)
    with col1:
        variable = st.selectbox("Selecciona una variable", numeric_cols)
    with col2:
        bins = st.slider("Número de bins", 5, 50, 20)
    fig = px.histogram(df, x=variable, nbins=bins)
    st.plotly_chart(fig, use_container_width=True)

elif opcion == "Modelo de Clasificación":
    st.subheader("Entrenamiento de Random Forest")
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    forest.fit(X_train, y_train)
    scores = cross_val_score(forest, X_train, y_train, cv=5)
    st.write(f"Precisión promedio (5-CV): **{scores.mean():.2f}**")

    y_pred = forest.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Precisión en test: **{acc:.2f}**")

    st.write("Matriz de Confusión")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicción", y="Real"), color_continuous_scale='Blues')
    st.plotly_chart(fig_cm)

    st.text("Reporte de Clasificación")
    st.text(classification_report(y_test, y_pred))

elif opcion == "Importancia de Variables":
    st.subheader("Top 10 Variables más Relevantes")
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False).head(10)
    fig = px.bar(feat_imp[::-1], orientation='h', labels={'value': 'Importancia', 'index': 'Variable'})
    st.plotly_chart(fig)

elif opcion == "Predicciones":
    st.subheader("Comparativa de Predicciones vs Realidad")
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    forest.fit(X_train, y_train)
    muestras = X_test.sample(10, random_state=42)
    pred = forest.predict(muestras)
    resultado = muestras.copy()
    resultado['Predicción'] = pred
    resultado['Real'] = y_test.loc[muestras.index].values

    fig = px.scatter(resultado.reset_index(), x=resultado.index, y='Predicción', 
                     labels={'x': 'Cliente'}, color_discrete_sequence=['blue'], symbol_sequence=['circle'])
    fig.add_scatter(x=resultado.index, y=resultado['Real'], mode='markers', 
                    marker=dict(color='orange', symbol='x', size=10), name='Real')
    st.plotly_chart(fig)
    st.dataframe(resultado)
