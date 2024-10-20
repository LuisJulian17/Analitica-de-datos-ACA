import pandas as pd
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport

# Ruta local del archivo descargado
ruta_local = r'C:\Users\luisj\OneDrive\Escritorio\BigData\DataMining\Data\OnlineRetail.xlsx'  # Usa raw string

# Cargar los datos desde el archivo local
data = pd.read_excel(ruta_local)

# Generar el reporte de perfilado
profile = ProfileReport(data, title="Informe de Análisis de Datos - Online Retail", explorative=True)

# Guardar el reporte como archivo HTML
profile.to_file("informe_analisis_datos.html")

print("¡Reporte generado y guardado como 'informe_analisis_datos.html'!")

# Limpiar datos
data.dropna(subset=['Description', 'CustomerID'], inplace=True)  # Eliminar filas con descripciones o IDs de clientes nulos
data.drop_duplicates(inplace=True)  # Eliminar filas duplicadas
data['TotalRevenue'] = data['Quantity'] * data['UnitPrice']  # Calcular ingresos totales

# Análisis de ventas por país
country_sales = data['Country'].value_counts().head(10)

# Ingresos totales por producto
product_revenue = data.groupby('Description')['TotalRevenue'].sum().sort_values(ascending=False).head(10)

# Análisis de comportamiento de clientes
customer_sales = data.groupby('CustomerID')['TotalRevenue'].sum().sort_values(ascending=False).head(10)

# Tendencias de ventas por mes
data['Month'] = data['InvoiceDate'].dt.to_period('M')
monthly_sales = data.groupby('Month')['TotalRevenue'].sum()

# Guardar gráficos
plt.figure(figsize=(12, 6))

# Gráfico de ventas por país
plt.subplot(1, 2, 1)
country_sales.plot(kind='bar', title='Top 10 Países por Ventas', color='skyblue')
plt.ylabel('Número de Ventas')
plt.xticks(rotation=45)

# Gráfico de ingresos por producto
plt.subplot(1, 2, 2)
product_revenue.plot(kind='bar', title='Top 10 Productos por Ingresos', color='orange')
plt.ylabel('Ingresos Totales')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('grafico_ventas.png')

# Gráfico de tendencias de ventas por mes
plt.figure(figsize=(10, 5))
monthly_sales.plot(kind='line', title='Tendencias de Ventas por Mes', marker='o')
plt.ylabel('Ingresos Totales')
plt.xlabel('Mes')
plt.xticks(rotation=45)
plt.grid()
plt.savefig('grafico_tendencias_ventas.png')


# Generar un reporte de análisis separado en HTML
with open("reporte_analisis.html", "w") as f:
    f.write("<html><head><title>Reporte de Análisis de Ventas</title></head><body>")
    f.write("<h1>Reporte de Análisis de Ventas</h1>")
    
    # Ventas por país
    f.write("<h2>Ventas por País</h2>")
    f.write("<ul>")
    for country, sales in country_sales.items():
        f.write(f"<li>{country}: {sales} ventas</li>")
    f.write("</ul>")
    
    # Ingresos por producto
    f.write("<h2>Ingresos por Producto</h2>")
    f.write("<ul>")
    for product, revenue in product_revenue.items():
        f.write(f"<li>{product}: ${revenue:.2f} ingresos</li>")
    f.write("</ul>")
    
    # Clientes que más compran
    f.write("<h2>Clientes que más compran</h2>")
    f.write("<ul>")
    for customer_id, revenue in customer_sales.items():
        f.write(f"<li>Cliente ID {customer_id}: ${revenue:.2f} ingresos</li>")
    f.write("</ul>")
    
    # Tendencias de ventas
    f.write("<h2>Tendencias de Ventas por Mes</h2>")
    f.write("<img src='grafico_tendencias_ventas.png' alt='Tendencias de Ventas por Mes'><br>")
    
    # Gráfico de segmentación de clientes
    f.write("<h2>Segmentación de Clientes</h2>")
    f.write("<img src='grafico_segmentacion_clientes.png' alt='Segmentación de Clientes'><br>")

    f.write("</body></html>")

print("¡Reporte de análisis generado y guardado como 'reporte_analisis.html'!")

