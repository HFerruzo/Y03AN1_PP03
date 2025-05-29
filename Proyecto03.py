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

# 2. CARGAR LOS DATOS
df = pd.read_csv("ovas_clientes_sintetico_full.csv")
print("Filas y columnas:", df.shape)
df.head()