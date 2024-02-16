import numpy as np  # Para operaciones numéricas y manejo de arrays
import pandas as pd  # Para la manipulación y análisis de datos
from sklearn.model_selection import train_test_split  # Para dividir los datos en conjuntos de entrenamiento y prueba
from sklearn.preprocessing import StandardScaler  # Para la normalización de los datos
import matplotlib.pyplot as plt  # Para la visualización de datos
from ipywidgets import interact, FloatSlider  # Para crear interfaces interactivas en Jupyter notebooks


# Cargar los datos desde un archivo Excel
datosExcel = pd.read_excel('Real estate valuation data set.xlsx')

# observamos una fraccion de ellos
datosExcel.head(5)

# Seleccionar las características y la variables de las casas
caracteristicas = ['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']
X = datosExcel[caracteristicas].values
Y = datosExcel['Y house price of unit area'].values.reshape(-1, 1)

# Crear una figura y una matriz de subgráficos con 2 filas y 3 columnas
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# Títulos personalizados para cada gráfico
nombres_caracteristicas = ['Edad de la casa', 'Distancia a la estación MRT más cercana', 'Número de tiendas de conveniencia', 'Latitud', 'Longitud', 'Distribución de Precios']

# Llenar cada subgráfico con un gráfico de dispersión para cada característica
for i, ax in enumerate(axes.flat):
    if i < 5:  # Para las primeras 5 características
        ax.scatter(datosExcel[caracteristicas[i]], datosExcel['Y house price of unit area'], alpha=0.4, edgecolor='none')
        ax.set_xlabel(caracteristicas[i])
        ax.set_ylabel('Precio [dolares/$m^2$]')
    else:  # Para el sexto gráfico, mostramos la distribución de precios
        ax.hist(datosExcel['Y house price of unit area'], bins=20, color='skyblue', edgecolor='black')
        ax.set_xlabel('Precio [dolares/$m^2$]')
        ax.set_ylabel('Frecuencia')
    ax.set_title(nombres_caracteristicas[i])
    ax.grid(visible=True, alpha=0.2)

plt.tight_layout()
plt.show()

# funcion que Divide los datos en conjuntos de entrenamiento y prueba, y normaliza las características.
def dividir_y_normalizar_datos(X, Y):

    # Dividir los datos en conjuntos de entrenamiento y prueba
    # test_size=0.2 indica que el 20% de los datos se utilizará como conjunto de prueba
    # random_state asegura que la división sea reproducible
    X_entrenamiento, X_prueba, Y_entrenamiento, Y_prueba = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Inicializar el objeto StandardScaler
    escalador_X = StandardScaler()

    # Ajustar el escalador solo a los datos de entrenamiento y transformarlos
    # Esto calcula la media y desviación estándar de cada característica en el conjunto de entrenamiento
    # y luego utiliza estos valores para escalar el conjunto de entrenamiento
    X_entrenamiento_escalado = escalador_X.fit_transform(X_entrenamiento)

    # Transformar los datos de prueba utilizando la misma transformación aplicada a los datos de entrenamiento
    # Es importante no ajustar el escalador con los datos de prueba para evitar el sesgo
    X_prueba_escalado = escalador_X.transform(X_prueba)

    # Añadir una columna de unos al inicio de las matrices escaladas para el término de sesgo (intercepto)

    X_entrenamiento_escalado = np.hstack([np.ones((X_entrenamiento_escalado.shape[0], 1)), X_entrenamiento_escalado])
    X_prueba_escalado = np.hstack([np.ones((X_prueba_escalado.shape[0], 1)), X_prueba_escalado])
    return X_entrenamiento_escalado, X_prueba_escalado, Y_entrenamiento, Y_prueba


def calcular_hipotesis(X, theta):
    """
    Calcula la hipótesis h_𝜃(X) para un conjunto de datos X y parámetros 𝜃.
    """
    return np.dot(X, theta)


def calcular_costo(X, Y, theta):
    """
    Calcula el valor de la función de costo J(𝜃).
    """
    m = len(Y)
    h = calcular_hipotesis(X, theta)
    return (1/(2*m)) * np.sum(np.square(h - Y))

def calcular_gradiente(X, Y, theta):
    """
    Calcula el gradiente de la función de costo.
    """
    m = len(Y)
    h = calcular_hipotesis(X, theta)
    return (1/m) * np.dot(X.T, (h - Y))

def actualizar_parametros(theta, tasa_aprendizaje, gradiente):
    """
    Actualiza los parámetros 𝜃 utilizando el gradiente de la función de costo.
    """
    return theta - tasa_aprendizaje * gradiente


def main():
    X_entrenamiento_escalado, X_prueba_escalado, Y_entrenamiento, Y_prueba = dividir_y_normalizar_datos(X, Y)

    theta_inicial = np.zeros((X_entrenamiento_escalado.shape[1], 1))

    tasa_aprendizaje = 0.01
    iteraciones = 1000

    theta = theta_inicial
    for i in range(iteraciones):
        gradiente = calcular_gradiente(X_entrenamiento_escalado, Y_entrenamiento, theta)
        theta = actualizar_parametros(theta, tasa_aprendizaje, gradiente)

    # Ejemplo de cómo se podrían imprimir los resultados del entrenamiento
    costo_final = calcular_costo(X_entrenamiento_escalado, Y_entrenamiento, theta)
    print("Costo final después del entrenamiento:", costo_final)
    print("Parámetros theta finales:", theta)


if __name__ == "__main__":
    main()
