import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nbformat as nbf
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import itertools

# Crear carpeta de reporte
output_folder = "reporte"
os.makedirs(output_folder, exist_ok=True)

# **1. Cargar los datos**
ruta_local = "Casos_COVID19_Limpio_Fase2.csv"
data = pd.read_csv(ruta_local, low_memory=False)
print("Columnas disponibles en el CSV:", data.columns.tolist())

# **2. Limpieza de datos**
data.columns = data.columns.str.strip().str.lower()
columnas_requeridas = ['nombre municipio', 'edad', 'unidad de medida de edad', 'sexo', 
                       'tipo de contagio', 'ubicación del caso', 'estado', 'recuperado', 
                       'fecha de muerte', 'tipo de recuperación']
data = data[columnas_requeridas]

# Convertir a tipos de datos adecuados
data['edad'] = pd.to_numeric(data['edad'], errors='coerce')
data['sexo'] = data['sexo'].str.upper().str.strip()
data['tipo de contagio'] = data['tipo de contagio'].str.title().str.strip()
data['estado'] = data['estado'].str.title().str.strip()
data['recuperado'] = data['recuperado'].str.title().str.strip()

# Manejo de valores nulos
data.dropna(subset=['edad', 'estado'], inplace=True)
data['fecha de muerte'] = data['fecha de muerte'].fillna('No reportado')
data.fillna('Desconocido', inplace=True)
data.drop_duplicates(inplace=True)

# Guardar datos limpios
data.to_csv(os.path.join(output_folder, "Casos_COVID19_Limpio_Fase2.csv"), index=False)
print("Datos limpios guardados.")

# **3. Generar informe de perfilado**
perfil = ProfileReport(data, title="Análisis COVID-19", explorative=True)
perfil.to_file(os.path.join(output_folder, "informe_perfilado_Fase3.html"))

# **4. Análisis Descriptivo y Visualizaciones**
sns.set(style="whitegrid")

def guardar_grafico(fig, filename):
    fig.savefig(os.path.join(output_folder, filename))
    plt.close(fig)

# Histograma de edades
fig = plt.figure(figsize=(8, 5))
sns.histplot(data["edad"], bins=30, kde=True, color="blue")
plt.title("Distribución de Edades")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
guardar_grafico(fig, "distribucion_edades_Fase3.png")

# Estado de los pacientes
estado_counts = data["estado"].value_counts()
fig = plt.figure(figsize=(8, 5))
sns.barplot(x=estado_counts.index, y=estado_counts.values, palette="viridis")
plt.title("Estado de los Pacientes")
plt.xlabel("Estado")
plt.ylabel("Número de Casos")
plt.xticks(rotation=45)
guardar_grafico(fig, "estado_pacientes_Fase3.png")

# Matriz de correlación
fig = plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Matriz de Correlación")
guardar_grafico(fig, "matriz_correlacion_Fase3.png")

# **5. Modelado Predictivo**
data_ml = data[['edad', 'sexo', 'tipo de contagio', 'estado']].copy()
data_ml = pd.get_dummies(data_ml, columns=['sexo', 'tipo de contagio'])

X = data_ml.drop(columns=['estado'])
y = data_ml['estado']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
fig = plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
guardar_grafico(fig, "matriz_confusion_Fase4.png")

# Importancia de características
feature_importances = model.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(feature_importances)[::-1]

fig = plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importances[sorted_idx], y=feature_names[sorted_idx], palette="coolwarm")
plt.title("Importancia de las Características")
plt.xlabel("Importancia")
plt.ylabel("Características")
guardar_grafico(fig, "importancia_caracteristicas_Fase4.png")

# **6. Generar Informe HTML**
tabla_html = data.describe().to_html()
html_reporte = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Informe de Análisis de Datos - COVID-19</title>
</head>
<body>
    <h1>Informe de Análisis de Datos - COVID-19</h1>
    <h2>Resumen Estadístico</h2>
    {tabla_html}
    <h2>Gráficos</h2>
    <img src='distribucion_edades_Fase3.png'>
    <img src='estado_pacientes_Fase3.png'>
    <img src='matriz_correlacion_Fase3.png'>
    <img src='matriz_confusion_Fase4.png'>
    <img src='importancia_caracteristicas_Fase4.png'>
</body>
</html>
"""

with open(os.path.join(output_folder, "informe_completo_Fase4.html"), "w", encoding="utf-8") as f:
    f.write(html_reporte)

print("¡Análisis completado! Todos los archivos generados se encuentran en la carpeta 'reporte'.")
