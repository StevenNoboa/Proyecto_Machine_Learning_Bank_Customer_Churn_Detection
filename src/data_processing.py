import pandas as pd
import numpy as np

# Carga de datos
df = pd.read_csv("../data_raw/Customer-Churn-Records.csv")

# Eliminación de columnas no necesarias
df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)
df.drop(columns=["Complain"], inplace=True)

# Mapeo de la variable "Gender"
gender_mapping = {'Male': 0, 'Female': 1}
df["Gender"] = df["Gender"].map(gender_mapping)

# Mapeo de la variable "Age" a categorías
age_mapping = {
    tuple(np.arange(18, 26)): 0,
    tuple(np.arange(26, 36)): 1,
    tuple(np.arange(36, 46)): 2,
    tuple(np.arange(46, 61)): 3,
    tuple(np.arange(61, 100)): 4
}
df['Age Category'] = df['Age'].map(lambda x: next((v for r, v in age_mapping.items() if x in r), x))

# Mapeo de la variable "CreditScore" a categorías
credit_score_mapping = {
    tuple(np.arange(350, 451)): 0,
    tuple(np.arange(451, 551)): 1,
    tuple(np.arange(551, 651)): 2,
    tuple(np.arange(651, 751)): 3,
    tuple(np.arange(751, 851)): 4
}
df['CreditScore Category'] = df['CreditScore'].map(lambda x: next((v for r, v in credit_score_mapping.items() if x in r), x))

# Mapeo de la variable "NumOfProducts"
NumOfProducts_mapping = {1: 0, 2: 0, 3: 1, 4: 1}
df['NumOfProducts'] = df['NumOfProducts'].map(NumOfProducts_mapping)

# Mapeo de la variable "Geography"
geography_mapping = {'France': 0, 'Spain': 1, 'Germany': 2}
df['Geography'] = df['Geography'].map(geography_mapping)

# Mapeo de la variable "Card Type"
card_type_mapping = {'DIAMOND': 0, 'GOLD': 1, 'PLATINUM': 2, 'SILVER': 3}
df['Card Type'] = df['Card Type'].map(card_type_mapping)

# Eliminación de columnas "CreditScore" y "Age"
df.drop(columns=["CreditScore", "Age"], inplace=True)

# Reorganización en Train y Test y guardado de datos procesados
df = df.assign(Exited=df.pop('Exited'))
df.to_csv("../data_processed/Churn_processed.csv", index=False)

df_train = df.iloc[:8000, :]
df_train.to_csv("../data_processed/Train_Churn_processed.csv", index=False)

df_test = df.iloc[8000:, :]
df_test.to_csv("../data_processed/Test_Churn_processed.csv", index=False)