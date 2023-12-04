import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

# Carga de datos
df_train = pd.read_csv("../data_processed/Train_Churn_processed.csv")

# División de datos en conjuntos de entrenamiento y prueba
X = df_train.drop(columns=["Exited"])
y = df_train["Exited"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Submuestreo de datos de entrenamiento
rus = RandomUnderSampler(random_state=10)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

# Carga del modelo entrenado
filename = '../models/finished_model_gs'
with open(filename, 'rb') as archivo_entrada:
    modelo_importado_1 = pickle.load(archivo_entrada)

# Entrenamiento del modelo con datos submuestreados
modelo_importado_1.fit(X_train_resampled, y_train_resampled)

# Predicciones y evaluación del modelo
y_pred = modelo_importado_1.predict(X_test)
