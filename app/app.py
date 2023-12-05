import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
from data_processing import gender_mapping, NumOfProducts_mapping, geography_mapping,card_type_mapping
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, roc_auc_score, f1_score,roc_curve, precision_recall_curve

st.set_page_config(page_title="Bank Customer churn Detection 📊", page_icon="🏬")
seleccion = st.sidebar.selectbox("Selecciona menu", ['Planteamiento', 'Data Science', 'Clientes'])

if seleccion == "Planteamiento":
    st.title("Bank Customer Churn Detection")
    img = Image.open("../img/img3.png")
    st.image(img)
    with st.expander("Introducción"):
        st.write("Resulta más costoso atraer a un nuevo cliente (por ejemplo, mayor gasto en marketing) que mantener a uno existente. Desde este punto de partida, el banco quiere conocer qué clientes pueden cancelar su cuenta y qué lleva a un cliente a tomar la decisión de abandonar la empresa. A largo plazo, este aspecto adquiere gran relevancia para el crecimiento del negocio.")
    with st.expander("Problemática"):
        st.write("### Tasa de Abandono (Churn Rate):")
        st.write("El Banco cuenta con una tasa de abandono del 20.38% según sus registros de cancelación de cuentas por parte de los clientes. Lo cual expone:")
        st.write("- Una tasa de abandono de clientes cercana al 25% o por encima se considera alta, indicando la necesidad de centrarse más en la retención de clientes.")
        st.write("- La pérdida significativa de clientes puede conducir a una disminución considerable en los ingresos, señalando un problema serio.")
        st.write("- En servicios financieros, como bancos o tarjetas de crédito, una tasa de abandono anual superior al 2-3% podría ser motivo de preocupación.")
    with st.expander("Acciones"):
        st.write("* Analizar y predecir el comportamiento del cliente, especialmente si es probable que abandonen el banco.")
        st.write("* Identificar factores que contribuyen a la rotación y ayudar en el desarrollo de estrategias para retener a los clientes.")

elif seleccion == "Data Science":
    st.title("Presentación Equipo Data Science")
    img1 = Image.open("../img/img2.png")
    st.image(img1)

    with st.expander("1 Datos empleados"):
        st.title("Dataset inicial")
        df = pd.read_csv("../data_raw/Customer-Churn-Records.csv")
        st.write(df.head())
        
        buffer = StringIO()
        df.info(buf=buffer)
        
    with st.expander("1.1 Información detallada"):
        st.text(buffer.getvalue())
        st.write("### Detalles del Conjunto de Datos:")
        st.write("Número de Entradas y Columnas:")
        st.write("  - El conjunto de datos tiene 10,000 entradas (filas) y 18 columnas.")
        st.write("Tipos de Datos:")
        st.write("  - La mayoría de las columnas contienen datos numéricos (int64 y float64).")
        st.write("  - Algunas columnas contienen datos categóricos representados como objetos (object).")
        st.write("Datos Faltantes:")
        st.write("  - No hay datos faltantes en ninguna de las columnas. Todas las columnas tienen 10,000 valores no nulos, lo que sugiere un conjunto de datos completo.")

    with st.expander("1.2 Resumen Estadístico"):
        st.write(df.describe())
        st.write("Edad (Age):")
        st.write("- La edad promedio de los clientes es de aproximadamente 38-39 años, con una dispersión relativamente baja (desviación estándar).")
        st.write("Puntaje de Crédito (CreditScore):")
        st.write("  - El puntaje de crédito promedio es de alrededor de 650, con una variabilidad moderada (desviación estándar).")
        st.write("  - El puntaje mínimo es 350 y el máximo es 850.")
        st.write("Balance:")
        st.write("  - El saldo promedio en cuentas es de aproximadamente 76,485.89, con una desviación estándar considerable.")
        st.write("  - Al menos el 25% de los clientes tienen un saldo de 0.")
        st.write("Productos Bancarios (NumOfProducts):")
        st.write("  - El número promedio de productos bancarios que poseen los clientes es de alrededor de 5.")
        st.write("  - Al menos el 25% de los clientes tiene 3 productos bancarios.")
        st.write("Tarjetas de Crédito (HasCrCard):")
        st.write("  - El 75% de los clientes tienen al menos una tarjeta de crédito.")
        st.write("Clientes Activos (IsActiveMember):")
        st.write("  - Alrededor del 51.5% de los clientes son miembros activos.")
        st.write("Salario Estimado (EstimatedSalary):")
        st.write("  - El salario estimado promedio es de aproximadamente 100,090.24.")
        st.write("Churn (Exited):")
        st.write("  - El 20.38% de los clientes han abandonado el banco.")
        st.write("Satisfacción (Satisfaction Score):")
        st.write("  - El puntaje promedio de satisfacción es de aproximadamente 3.01, con un rango de 1 a 5.")
        st.write("Tipo de Tarjeta (Card Type):")
        st.write("  - Hay cuatro tipos de tarjetas de crédito ('Card Type').")
        st.write("Puntos Ganados (Point Earned):")
        st.write("  - El número promedio de puntos ganados es de aproximadamente 606.52, con un máximo de 1000.")
    with st.expander("1.3 Distribución del Target 'Exited'"):
        colores = ['#ec6363', '#11e1a8']
        plt.figure(figsize=(3, 3))
        plt.pie(df['Exited'].value_counts(), autopct='%1.2f%%', colors=colores)
        plt.title('Distribución del Target "Exited"')
        plt.legend(['No Exited', 'Exited'], loc='upper right')
        st.pyplot(plt)
        exited_situacion = pd.DataFrame(df["Exited"].value_counts())
        st.table(exited_situacion)
        st.write("El Target **Exited** está desbalanceado. Tenemos más registros de clientes que no han salido del banco.")
    with st.expander("1.4 Visualizaciones iniciales"):
        fig, axis = plt.subplots(9, 2, figsize=(20, 35))
        No_Exited = df[df['Exited'] == 0]
        Exited = df[df['Exited'] == 1]
        axes = axis.ravel()  
        for i in range(len(df.columns)):
            axes[i].hist(No_Exited.iloc[:, i], bins=40, color='r', alpha=0.4)
            axes[i].hist(Exited.iloc[:, i], bins=40, color='g', alpha=0.4)
            axes[i].set_title(df.columns[i]) 
        axes[0].legend(['No_Exited', 'Exited'])
        st.pyplot(fig)

        fig = plt.figure(figsize=(15,15))
        sns.heatmap(df.corr(numeric_only=True),
        vmin=-1,
        vmax=1,
        cmap=sns.color_palette("vlag", as_cmap=True),
        annot=True)
        st.pyplot(fig)

    with st.expander("2 Transformaciones"):
        st.write("### Columnas Eliminadas:")
        st.write("- 'RowNumber', 'CustomerId', 'Surname'")
        st.write("- 'Complain'")
        
        st.write("### Mapeo de la variable 'Gender':")
        st.write(gender_mapping)
        st.write("### Mapeo de la variable 'Age' a categorías:")
        age_mapping = {
                    str(tuple(np.arange(18, 26))): 0,
                    str(tuple(np.arange(26, 36))): 1,
                    str(tuple(np.arange(36, 46))): 2,
                    str(tuple(np.arange(46, 61))): 3,
                    str(tuple(np.arange(61, 100))): 4}
        
        st.write(age_mapping)
        
        st.write("### Mapeo de la variable 'CreditScore' a categorías:")
        credit_score_mapping = {
                    str(tuple(np.arange(350, 451))): 0,
                    str(tuple(np.arange(451, 551))): 1,
                    str(tuple(np.arange(551, 651))): 2,
                    str(tuple(np.arange(651, 751))): 3,
                    str(tuple(np.arange(751, 851))): 4
}
        st.write(credit_score_mapping)
        
        st.write("### Mapeo de la variable 'NumOfProducts':")
        st.write(NumOfProducts_mapping)
        
        st.write("### Mapeo de la variable 'Geography':")
        st.write(geography_mapping)
        
        st.write("### Mapeo de la variable 'Card Type':")
        st.write(card_type_mapping)
        
        st.write("### Columnas Eliminadas:")
        st.write("- 'CreditScore', 'Age'")
        
        st.write("### Guardado de Datos Procesados: Dataset Final")
        st.write("- 'Churn_processed.csv'")
        df = pd.read_csv("../data_processed/Churn_processed.csv")
        st.write(df.head())
        st.write("Dividimos el Dataset Final en:")
        st.write("- 'Train_Churn_processed.csv': (8000, 14) ")
        st.write("- 'Test_Churn_processed.csv': (2000, 14)")
    with st.expander("2.1 Visualizaciones finales"):
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(22, 22))
        columns_for_analysis = df.columns[df.columns != 'Exited']
        for i, column in enumerate(columns_for_analysis):
            ax = axes[i // 4, i % 4]
            sns.kdeplot(data=df, x=column, hue='Exited', fill=True, common_norm=False, ax=ax)
            ax.set_title(f'{column} vs. Exited')
        st.pyplot(fig)

        fig = plt.figure(figsize=(20,20))
        sns.heatmap(df.corr(numeric_only= True),
           vmin=-1,
           vmax=1,
           cmap=sns.color_palette("vlag", as_cmap=True),
           annot=True)
        st.pyplot(fig)
    with st.expander("3 Modelos"):
        st.title("Configuración del Modelo")
        st.write("Modelos a Emplear:")
        st.write("   1. Logistic Regression")
        st.write("   2. Random Forest Classifier")
        st.write("   3. GradientBoostingClassifier")
        st.write("   4. KNeighborsClassifier (KNN)")
        st.write("   5. Support Vector Machines (SVM): SVC")
        st.write("Enfoque en el Scoring de 'Recall':")
        st.write("   1. Mide la proporción de instancias positivas correctamente clasificadas entre todas las instancias que realmente son positivas.")
        st.write("   2. En este caso, el costo de los falsos negativos es alto y queremos identificar la mayor cantidad posible de casos positivos.")
    with st.expander("3.1 Solución al desbalance de los datos"): 
        df_train = pd.read_csv("../data_processed/Train_Churn_processed.csv")
        X = df_train.drop(columns=["Exited"])
        y = df_train["Exited"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
        colores = ['#ec6363', '#11e1a8']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].pie(y.value_counts(), autopct='%1.1f%%', colors=colores)
        axes[0].set_title('Distribución del Target en y')
        axes[0].legend(['No Exited', 'Exited'], loc='upper right')

        axes[1].pie(y_train.value_counts(), autopct='%1.1f%%', colors=colores)
        axes[1].set_title('Distribución del Target en y_train')
        axes[1].legend(['No Exited', 'Exited'], loc='upper right')

        axes[2].pie(y_test.value_counts(), autopct='%1.1f%%', colors=colores)
        axes[2].set_title('Distribución del Target en y_test')
        axes[2].legend(['No Exited', 'Exited'], loc='upper right')
        st.pyplot(fig)
        st.write("Aplicamos Submuestreo (Undersampling):")
        st.write("- Esta estrategia implica reducir la cantidad de instancias de la clase mayoritaria (en este caso, 0) para igualarla con la cantidad de instancias de la clase minoritaria (en este caso, 1).")
        rus = RandomUnderSampler(random_state=10)
        X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
        colores = ['#ec6363', '#11e1a8']
        fig = plt.figure(figsize=(4,4))
        plt.pie(y_train_resampled.value_counts(),
                autopct='%1.2f%%',
                colors=colores)
        plt.title('Undersumpling: Distribución del Target "Exited"')
        plt.legend(['No Exited', 'Exited'], loc='upper right')
        st.pyplot(fig)
        exited_situacion = pd.DataFrame(y_train_resampled.value_counts())
        st.table(exited_situacion)
    
    with st.expander("3.2 Pipeline y prámetros empleados"):
        pipe = pd.read_csv("../data_csv/pipeline.csv")
        parametros = pd.read_csv("../data_csv/parametros.csv")
        st.write(pipe)
        st.write(parametros)

        st.write("Una vez generado el pipeline y los parámetros de espacio de búsqueda, introducimos estos en un:")
        st.write("- clf_gs.GreadSearchCV:estimator=pipe, param_grid=search_space, cv=5, scoring='accuracy', verbose=3, n_jobs=-1")
        st.write("Posteriormente, lo entrenamos con los datos de Undersumpling:")
        st.write("- clf_gs.fit(X_train_resampled, y_train_resampled)")
        st.write("Obtenemos:")
        best_classifier_info = "- Mejor clasificador: Pipeline(steps=[('scaler', StandardScaler()), ('selectkbest', SelectKBest(k=9)), ('classifier', RandomForestClassifier(max_depth=5))])"
        st.write(best_classifier_info)
        recall_score_info = "- Mejor puntuación de recall: 0.7483124986824631"
        st.write(recall_score_info)
        best_params_info = "- Mejores parámetros: {'classifier': RandomForestClassifier(), 'classifier__max_depth': 5, 'classifier__n_estimators': 100, 'scaler': StandardScaler(), 'selectkbest__k': 9}"
        st.write(best_params_info)

    with st.expander("4 Evaluación y resultados en el Test de Train_Churn"):
        filename = '../models/finished_model_gs'
        with open(filename, 'rb') as archivo_entrada:
            modelo_importado_1 = pickle.load(archivo_entrada)
        
        modelo_importado_1.fit(X_train_resampled, y_train_resampled)
        y_pred = modelo_importado_1.predict(X_test)

        st.write("Accuracy Score:", accuracy_score(y_test, y_pred))
        st.write("Precision Score:", precision_score(y_test, y_pred))
        st.write("Recall Score:", recall_score(y_test, y_pred))
        st.write("F1 Score:", f1_score(y_test, y_pred))
        st.write("ROC AUC Score:", roc_auc_score(y_test, y_pred))
        st.write("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        conf_matrix_test = confusion_matrix(y_test, y_pred, normalize='true')
        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix_test, annot=True, fmt=".2%", cmap='Reds', xticklabels=['No_Exited', 'Exited'], yticklabels=['No_Exited', 'Exited'])
        plt.title('Matriz de Confusión Normalizada')
        st.pyplot(fig)

        y_pred = modelo_importado_1.predict(X_test)
        y_proba = modelo_importado_1.predict_proba(X_test)[:, 1]  

        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)


        fig = plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        st.pyplot(fig)

        fig = plt.figure(figsize=(8, 8))
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        st.pyplot(fig)
    with st.expander("4.1 Evaluación y resultados en el Test_Churn"):
        df_test = pd.read_csv("../data_processed/Test_Churn_processed.csv")
        st.write(df_test.head())
        X_t = df_test.drop(columns=["Exited"])
        y_t = df_test["Exited"]
        # Realizar predicciones
        y_pred_test = modelo_importado_1.predict(X_t)

        # Evaluar el rendimiento del modelo
        st.write("Accuracy Score:", accuracy_score(y_t, y_pred_test))
        st.write("Precision Score:", precision_score(y_t, y_pred_test))
        st.write("Recall Score:", recall_score(y_t, y_pred_test))
        st.write("F1 Score:", f1_score(y_t, y_pred_test))
        st.write("ROC AUC Score:", roc_auc_score(y_t, y_pred_test))
        st.write("Confusion Matrix:\n", confusion_matrix(y_t, y_pred_test))

        # Visualizaciones
        conf_matrix_test = confusion_matrix(y_t, y_pred_test, normalize='true')
        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix_test, annot=True, fmt=".2%", cmap='Reds', xticklabels=['No_Exited', 'Exited'], yticklabels=['No_Exited', 'Exited'])
        plt.title('Matriz de Confusión')
        st.pyplot(fig)

        # Curva ROC
        fpr, tpr, thresholds = roc_curve(y_t, modelo_importado_1.predict_proba(X_t)[:, 1])
        fig = plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        st.pyplot(fig)
        # Curva Precision-Recall
        precision, recall, _ = precision_recall_curve(y_t, modelo_importado_1.predict_proba(X_t)[:, 1])
        fig = plt.figure(figsize=(8, 8))
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        st.pyplot(fig)

    with st.expander("5 Impacto de las variables"):
        features_importance_df = pd.read_csv("../data_csv/features_importance.csv")
        fig= plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=features_importance_df, palette='viridis')
        plt.title('Importancia de las Características Seleccionadas')
        plt.xlabel('Importancia')
        plt.ylabel('Característica')
        st.pyplot(fig)

        st.write(features_importance_df)
        st.write("""
        ### Importancia de las variables en el Modelo
        1. **Categoría de Edad (Age Category):** Con una importancia del 42.64%, la categoría de edad es la característica más influyente. Esto significa que la edad tiene un fuerte impacto en la capacidad del modelo para hacer predicciones.
        2. **Número de Productos (NumOfProducts):** Con una importancia del 20.93%, el número de productos también es una característica clave en el modelo. Indica que la cantidad de productos bancarios que posee un cliente afecta significativamente a las predicciones del modelo.
        3. **Miembro Activo (IsActiveMember):** Con una importancia del 13.15%, ser un miembro activo también es una característica influyente. Indica que la actividad de membresía tiene un impacto en las predicciones del modelo.
        4. **Geografía (Geography):** Con una importancia del 7.85%, la geografía también juega un papel importante en el modelo. La ubicación geográfica de los clientes afecta las predicciones del modelo.
        5. **Saldo (Balance):** Con una importancia del 6.93%, el saldo en las cuentas también es una característica relevante. Indica que la cantidad de dinero en la cuenta de un cliente es informativa para el modelo.
        6. **Salario Estimado (EstimatedSalary):** Con una importancia del 3.14%, el salario estimado también contribuye al modelo, aunque en menor medida que otras características.
        7. **Género (Gender):** Con una importancia del 3.11%, el género también tiene cierta influencia en las predicciones del modelo.
        8. **Categoría de Puntaje de Crédito (CreditScore Category):** Con una importancia del 1.16%, la categoría de puntaje de crédito también contribuye al modelo, pero en menor medida.
        9. **Tipo de Tarjeta (Card Type):** Con una importancia del 1.08%, el tipo de tarjeta es la característica menos influyente en el modelo.
        """)
    with st.expander("6 Conclusiones"):
        st.title("Conclusiones generales del análisis y modelado.")
        ## Demografía
        st.header("1. Demografía:")
        st.write("- La edad y la geografía son factores críticos en el abandono del servicio.")
        st.write("- La mayoría de los clientes tiene entre 26 y 44 años.")

        # Productos y Finanzas
        st.header("2. Productos y Finanzas:")
        st.write("- La mayoría tiene 1-2 productos bancarios. Sin embargo, la tasa de abandono es mayor con los que cuentan con 3-4 productos. El saldo promedio es de 76,485.89.")

        # Tarjetas de Crédito
        st.header("3. Tarjetas de Crédito:")
        st.write("- La mayoría tiene al menos una tarjeta de crédito.")
        st.write("- Cuatro tipos de tarjetas, siendo DIAMOND la más común.")

        # Churn y Actividad
        st.header("4. Churn y Actividad:")
        st.write("- 20.38% de abandono.")
        st.write("- Cerca del 51.5% son miembros activos.")
        st.write("- Satisfacción promedio: 3.01 de 5.")

        # Modelo Predictivo
        st.header("5. Modelo Predictivo:")
        st.write("- Edad y número de productos clave en predicciones.")
        st.write("- Análisis destaca la influencia significativa de la edad y productos.")

elif seleccion == "Clientes":
    # Cargar datos procesados
    df_test = pd.read_csv("../data_processed/Test_Churn_processed.csv")
    # Dividir datos en características (X) y objetivo (y)
    X_t = df_test.drop(columns=["Exited"])
    y_t = df_test["Exited"]

    # Cargar modelo
    filename = '../models/finished_model_gs'
    with open(filename, 'rb') as archivo_entrada:
        modelo_importado_1 = pickle.load(archivo_entrada)
    
    y_pred_test=modelo_importado_1.predict(X_t)

    # Función para hacer predicciones
    def hacer_predicciones(features):
        return modelo_importado_1.predict(features)

    # Streamlit Cliente
    st.title("Negocio: Aplicación de Retención de Clientes")
    img = Image.open("../img/img1.png")
    st.image(img)
    # Sidebar con opciones
    st.sidebar.header("Opciones")
    show_data = st.sidebar.checkbox("Mostrar datos procesados por Data Science")
    st.sidebar.header("Pruebe el modelo:")
    feature_columns = st.sidebar.multiselect("Seleccionar columnas", X_t.columns)

    # Mostrar datos procesados si se selecciona
    if show_data:
        st.subheader("Datos Procesados")
        st.write(df_test)
    user_data_dict = {}
    for column in feature_columns:
        st.subheader(f"Ingrese valores para la columna {column}")

        if column == "Geography":
            st.write("0 : Francia, 1 : España, 2 : Alemania")
            entered_value = st.number_input("Ingrese el valor", min_value=0, max_value=2, step=1, key=f"{column}_input")
        elif column == "Gender":
            st.write("0 : Masculino, 1 : Femenino")
            entered_value = st.number_input("Ingrese el valor", min_value=0, max_value=1, step=1, key=f"{column}_input")
        elif column == "Tenure":
            st.write("Antigüedad del cliente")
            entered_value = st.number_input("Ingrese el valor", min_value=0, max_value=10, step=1, key=f"{column}_input")
        elif column == "Balance":
            st.write("Saldo del cliente")
            entered_value = st.number_input("Ingrese el valor", min_value=0, key=f"{column}_input")
        elif column == "NumOfProducts":
            st.write("0 : 1 o 2 productos, 1 : 3 o 4 productos")
            entered_value = st.number_input("Ingrese el valor", min_value=0, max_value=1, step=1, key=f"{column}_input") 
        elif column == "HasCrCard":
            st.write("0 : No tiene tarjeta de crédito ,  1 : Sí tiene tarjeta de crédito")
            entered_value = st.number_input("Ingrese el valor", min_value=0, max_value=1, step=1, key=f"{column}_input")         
        elif column == "IsActiveMember":
            st.write("0 : No es miembro activo, 1 : Es miembro activo")
            entered_value = st.number_input("Ingrese el valor", min_value=0, max_value=1, step=1, key=f"{column}_input")
        elif column == "EstimatedSalary":
            st.write("Salario estimado del cliente")
            entered_value = st.number_input("Ingrese el valor", min_value=0, key=f"{column}_input")
        elif column == "Satisfaction Score":
            st.write("Satisfacción del cliente: valores del 1 al 5")
            entered_value = st.number_input("Ingrese el valor", min_value=1, max_value=5, step=1, key=f"{column}_input")
        elif column == "Card Type":
            st.write("0 : DIAMOND, 1 : GOLD, 2 : PLATINUM, 3 : SILVER")
            entered_value = st.number_input("Ingrese el valor", min_value=0, max_value=3, step=1, key=f"{column}_input")
        elif column == "Point Earned":
            st.write("Puntos obtenidos por el cliente")
            entered_value = st.number_input("Ingrese el valor", min_value=0, key=f"{column}_input")
        elif column == "Age Category":
            st.write("0 : 18-25, 1 : 26-35, 2 : 36-45, 3 : 46-60, 4 : 61-99")
            entered_value = st.number_input("Ingrese el valor", min_value=0, max_value=4, step=1, key=f"{column}_input")
        elif column == "CreditScore Category":
            st.write("Puntos que el banco asigna para medir la facilidad de obtener un crédito")
            st.write("0 : 350-450, 1 : 451-550, 2 : 551-650, 3 : 651-750, 4 : 751-850")
            entered_value = st.number_input("Ingrese el valor", min_value=0, max_value=4, step=1, key=f"{column}_input")
                
        user_data_dict[column] = entered_value

    user_data_df = pd.DataFrame([user_data_dict])

    st.write("Datos del Usuario:")
    st.write(user_data_df)

    # Realizar predicción
    if st.button("Realizar Predicción"):
        prediction = hacer_predicciones(user_data_df)
        st.success(f"Predicción: {prediction[0]}")

        st.write("## Tras la prueba, observeamos las métricas del modelo_final empleado:")
# Evaluación del modelo
        st.write("Accuracy Score:", accuracy_score(y_t, y_pred_test))
        st.write("Precision Score:", precision_score(y_t, y_pred_test))
        st.write("Recall Score:", recall_score(y_t, y_pred_test))
        st.write("F1 Score:", f1_score(y_t, y_pred_test))
        st.write("ROC AUC Score:", roc_auc_score(y_t, y_pred_test))
        st.write("Confusion Matrix:\n", confusion_matrix(y_t, y_pred_test))

# Visualización de la matriz de confusión
        conf_matrix_test = confusion_matrix(y_t, y_pred_test, normalize='true')
        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix_test, annot=True, fmt=".2%", cmap='Reds', xticklabels=['No_Exited', 'Exited'], yticklabels=['No_Exited', 'Exited'])
        plt.title('Matriz de Confusión')
        st.pyplot(fig)

        precision, recall, _ = precision_recall_curve(y_t, modelo_importado_1.predict_proba(X_t)[:, 1])
        fig = plt.figure(figsize=(8, 8))
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        st.pyplot(fig)

        features_importance_df = pd.read_csv("../data_csv/features_importance.csv")
        fig= plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=features_importance_df, palette='viridis')
        plt.title('Importancia de las Características Seleccionadas')
        plt.xlabel('Importancia')
        plt.ylabel('Característica')
        st.pyplot(fig)
    with st.expander("Conclusiones finales para el cliente"):
        st.title("Conclusiones Finales")
        # Reflexión sobre el abandono de clientes
        st.header("1. Clientes:")
        st.write("El análisis revela que el abandono de clientes puede estar relacionado con varios factores, destacando:")
        st.write("- La edad, donde clientes más jóvenes y mayores tienden a abandonar más.")
        st.write("- El número de productos bancarios, sugiriendo que una oferta personalizada podría retener a más clientes.")
        st.write("- La geografía, lo que podría indicar diferencias en la calidad del servicio en diversas ubicaciones.")

        # Estrategias de Retención
        st.header("2. Estrategias de Retención:")
        st.write("Con base en los hallazgos, posibles estrategias de retención podrían incluir:")
        st.write("- Programas personalizados para diferentes grupos de edad.")
        st.write("- Ofertas especiales para clientes con menos productos bancarios.")
        st.write("- Mejora de servicios en ubicaciones geográficas específicas.")

        # Mejora Continua
        st.header("3. Enfoque en la Mejora Continua:")
        st.write("La implementación de un modelo predictivo permite un enfoque proactivo para retener clientes.")
        st.write("Es vital seguir monitoreando y ajustando estrategias basadas en la evolución del comportamiento del cliente.")
