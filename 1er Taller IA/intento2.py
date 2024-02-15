import numpy as np  # Para operaciones numéricas y manejo de arrays
import pandas as pd  # Para la manipulación y análisis de datos
from sklearn.model_selection import train_test_split  # Para dividir los datos en conjuntos de entrenamiento y prueba
from sklearn.preprocessing import StandardScaler  # Para la normalización de los datos
import matplotlib.pyplot as plt  # Para la visualización de datos
from ipywidgets import interact, FloatSlider  # Para crear interfaces interactivas en Jupyter notebooks


# Cargar los datos desde un archivo Excel
datosExcel = pd.read_excel('Real estate valuation data set.xlsx')

# Seleccionar las características y la variables de las casas
caracteristicas = ['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']
X = datosExcel[caracteristicas].values
Y = datosExcel['Y house price of unit area'].values.reshape(-1, 1)

# funcion que Divide los datos en conjuntos de entrenamiento y prueba, y normaliza las características.
def dividir_y_normalizar_datos(X, Y):

    X_entrenamiento, X_prueba, Y_entrenamiento, Y_prueba = train_test_split(X, Y, test_size=0.2, random_state=42)
    escalador_X = StandardScaler()
    X_entrenamiento_escalado = escalador_X.fit_transform(X_entrenamiento)
    X_prueba_escalado = escalador_X.transform(X_prueba)
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
