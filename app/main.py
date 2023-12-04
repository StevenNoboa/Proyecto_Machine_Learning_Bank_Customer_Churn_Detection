import streamlit as st 
import pandas as pd
from PIL import Image
import streamlit.components.v1 as c
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, roc_auc_score

st.set_page_config(page_title="Bank Customer churn Detection", page_icon= ":bar_chart:")
df= pd.read_csv("../data_raw/Customer-Churn-Records.csv")

seleccion = st.sidebar.selectbox("Selecciona menu", ['Home','Datos',"Modelo"])

if seleccion == "Home":
    st.title("Bank Customer Churn Detection")
    img = Image.open("../img/img3.png")
    st.image(img)
    with st.expander("Introducción"):
        st.write("Resulta más costoso atraer a un nuevo cliente (por ejemplo, mayor gasto en marketing) que mantener a uno existente. Desde este punto de partida, el banco quiere conocer qué clientes puden cancelar su cuenta y qué lleva a un cliente a tomar la decisión de abandonar la empresa. A largo plazo, este aspecto adquiere gran relevancia para el crecimiento del negocio.")

    with st.expander("Dataset"):
        st.write("El Dataset con el que vamos a trabajar es el siguiente:")
        st.write("[Kaggle - Bank Customer Data for Customer Churn](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)")
        df= pd.read_csv("../data_raw/Customer-Churn-Records.csv")
        st.write(df.head())

elif seleccion == "Datos":
    st.title("Planteamiento")
    df= pd.read_csv("../data_raw/Customer-Churn-Records.csv")

    filtro = st.sidebar.selectbox("Selecciona país", df['Geography'].unique())
    df_filtered = df[df['Geography']==filtro]
    st.write(df_filtered)

    # file = open("heatmap.html", "r")
    # c.html(file.read(), height=400)
    # df_filtered.rename(columns={"latidtud": "latitud", "longitud": "longitud"}, inplace=True)

    # st.map(df_filtered)

    filtro_2 = st.sidebar.radio("Elige el país", [1,2,3])

elif seleccion == "Modelo":
    st.title("Prediciones")
    with open('../models/finished_model_gs', 'rb') as archivo_entrada:
        modelo_importado = pickle.load(archivo_entrada)
    df_test = pd.read_csv("../data_processed/Test_Churn_processed.csv")
    st.write("Prueba del modelo final en X_t (los registros de test para la prueba):")
    st.write(df_test.head())
            
    X_test = df_test.drop(columns=["Exited"])
    y_test = df_test["Exited"]
    y_pred_test = modelo_importado.predict(X_test)

    st.write("Predicciones del modelo:")
    st.write(modelo_importado.predict(X_test))
    st.write("Evaluación de las métricas del modelo:")
    st.write("accuracy_score", accuracy_score(y_pred_test, y_test))
    st.write("precision_score", precision_score(y_pred_test, y_test))
    st.write("recall_score", recall_score(y_pred_test, y_test))
    st.write("roc_auc_score", roc_auc_score(y_pred_test, y_test))
    st.write("confusion_matrix\n", confusion_matrix(y_pred_test, y_test))

    conf_matrix_test = confusion_matrix(y_pred_test, y_test, normalize= "true")
    st.write("Matriz de Confusión:")
    st.write("Visualización de la Matriz de Confusión:")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix_test, annot=True, fmt=".2%", cmap='Reds', xticklabels=['No_Exited', 'Exited'], yticklabels=['No_Exited', 'Exited'])
    plt.title('Matriz de Confusión')
    st.pyplot(fig)

    #streamlit run main.py