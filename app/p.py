import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la aplicación
st.title('App de Machine Learning con Streamlit')

# Cargar datos de ejemplo
@st.cache
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    data = pd.read_csv(url)
    return data

data = load_data()

# Sidebar para opciones de usuario
st.sidebar.header('Configuración de Modelo')
selected_model = st.sidebar.selectbox('Seleccione un modelo', ['Regresión Lineal', 'Random Forest'])

# Dividir datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data[['Pclass', 'Age', 'Fare']], data['Survived'], test_size=0.2, random_state=42)

# Entrenar modelos
if selected_model == 'Regresión Lineal':
    model = LinearRegression()
    model.fit(X_train, y_train)
elif selected_model == 'Random Forest':
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

# Opciones de usuario para ajustar valores
st.sidebar.header('Ajuste de Parámetros')
user_input_age = st.sidebar.slider('Seleccione la Edad:', float(data['Age'].min()), float(data['Age'].max()), float(data['Age'].mean()))
user_input_fare = st.sidebar.slider('Seleccione la Tarifa:', float(data['Fare'].min()), float(data['Fare'].max()), float(data['Fare'].mean()))

# Realizar predicciones
user_prediction = model.predict([[1, user_input_age, user_input_fare]])

# Visualizaciones
st.header('Visualizaciones')
st.subheader('Distribución de Edades en el Conjunto de Datos')
fig, ax = plt.subplots()
sns.histplot(data['Age'].dropna(), kde=True, ax=ax)
st.pyplot(fig)

# Predicción del Usuario
st.header('Predicción del Usuario')
st.write(f'**Edad Seleccionada:** {user_input_age}')
st.write(f'**Tarifa Seleccionada:** {user_input_fare}')
st.write(f'**Predicción del Modelo:** {user_prediction[0]}')