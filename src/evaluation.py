import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, roc_auc_score, f1_score,roc_curve, precision_recall_curve
import pickle

# Cargar datos de prueba
df_test = pd.read_csv("../data_processed/Test_Churn_processed.csv")

# Cargar el modelo previamente entrenado
filename = '../models/finished_model_gs'
with open(filename, 'rb') as archivo_entrada:
    modelo_importado_1 = pickle.load(archivo_entrada)

# Variables de entrada y salida
X_t = df_test.drop(columns=["Exited"])
y_t = df_test["Exited"]

# Realizar predicciones
y_pred_test = modelo_importado_1.predict(X_t)

# Evaluar el rendimiento del modelo
print("Accuracy Score:", accuracy_score(y_t, y_pred_test))
print("Precision Score:", precision_score(y_t, y_pred_test))
print("Recall Score:", recall_score(y_t, y_pred_test))
print("F1 Score:", f1_score(y_t, y_pred_test))
print("ROC AUC Score:", roc_auc_score(y_t, y_pred_test))
print("Confusion Matrix:\n", confusion_matrix(y_t, y_pred_test))

# Visualizaciones
conf_matrix_test = confusion_matrix(y_t, y_pred_test, normalize='true')
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_test, annot=True, fmt=".2%", cmap='Reds', xticklabels=['No_Exited', 'Exited'], yticklabels=['No_Exited', 'Exited'])
plt.title('Matriz de Confusión')
plt.show()

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_t, modelo_importado_1.predict_proba(X_t)[:, 1])
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Curva Precision-Recall
precision, recall, _ = precision_recall_curve(y_t, modelo_importado_1.predict_proba(X_t)[:, 1])
plt.figure(figsize=(8, 8))
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()




















