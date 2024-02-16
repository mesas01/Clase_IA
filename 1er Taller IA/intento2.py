import numpy as np  # Para operaciones numéricas y manejo de arrays
import pandas as pd  # Para la manipulación y análisis de datos
from sklearn.model_selection import train_test_split  # Para dividir los datos en conjuntos de entrenamiento y prueba
from sklearn.preprocessing import StandardScaler  # Para la normalización de los datos
import matplotlib.pyplot as plt  # Para la visualización de datos
from ipywidgets import interact, FloatSlider, IntSlider  # Para crear interfaces interactivas en Jupyter notebooks
from IPython.display import display

# Cargar los datos desde un archivo Excel
datosExcel = pd.read_excel('Real estate valuation data set.xlsx')

# Seleccionar las características y la variables de las casas
caracteristicas = ['X2 house age', 'X3 distance to the nearest MRT station',
                   'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']
X = datosExcel[caracteristicas].values
Y = datosExcel['Y house price of unit area'].values.reshape(-1, 1)


# funcion que Divide los datos en conjuntos de entrenamiento y prueba, y normaliza las características.
def dividir_y_normalizar_datos(X, Y):

    X_entrenamiento, X_prueba, Y_entrenamiento, Y_prueba = train_test_split(X, Y, test_size=0.2, random_state=34)
    escalador_X = StandardScaler()
    X_entrenamiento_escalado = escalador_X.fit_transform(X_entrenamiento)
    X_prueba_escalado = escalador_X.transform(X_prueba)
    X_entrenamiento_escalado = np.hstack([np.ones((X_entrenamiento_escalado.shape[0], 1)), X_entrenamiento_escalado])
    X_prueba_escalado = np.hstack([np.ones((X_prueba_escalado.shape[0], 1)), X_prueba_escalado])
    return X_entrenamiento_escalado, X_prueba_escalado, Y_entrenamiento, Y_prueba


# Calcula la hipótesis h_𝜃(X) para un conjunto de datos X y parámetros 𝜃.
def calcular_hipotesis(X, theta):

    return np.dot(X, theta)


# Calcula el valor de la función de costo J(𝜃).
def calcular_costo(X, Y, theta):

    m = len(Y)
    h = calcular_hipotesis(X, theta)
    return (1/(2*m)) * np.sum(np.square(h - Y))


# Calcula el gradiente de la función de costo.
def calcular_gradiente(X, Y, theta):

    m = len(Y)
    h = calcular_hipotesis(X, theta)
    return (1/m) * np.dot(X.T, (h - Y))


# Actualiza los parámetros 𝜃 utilizando el gradiente de la función de costo.
def actualizar_parametros(theta, tasa_aprendizaje, gradiente):
    return theta - tasa_aprendizaje * gradiente


def entrenar_y_evaluar_modelo(tasa_aprendizaje=0.01, iteraciones=1000):
    # Escalado significa la normalización de los datos
    X_entrenamiento_escalado, X_prueba_escalado, Y_entrenamiento, Y_prueba = dividir_y_normalizar_datos(X, Y)
    # Inicializa el vector de parámetros θ con ceros.
    theta_inicial = np.zeros((X_entrenamiento_escalado.shape[1], 1))
    theta = theta_inicial
    historial_costos_entrenamiento = []  # Lista para almacenar el costo por iteración

    # Entrenar el modelo utilizando el gradiente descendiente
    for i in range(iteraciones):
        gradiente = calcular_gradiente(X_entrenamiento_escalado, Y_entrenamiento, theta)
        theta = actualizar_parametros(theta, tasa_aprendizaje, gradiente)
        costo = calcular_costo(X_entrenamiento_escalado, Y_entrenamiento, theta)
        historial_costos_entrenamiento.append(costo)  # Guardar el costo de la iteración actual

    # Graficar la curva de aprendizaje utilizando el historial de costos
    plt.figure(figsize=(8, 5))
    plt.plot(historial_costos_entrenamiento, label='Costo de Entrenamiento')
    plt.xlabel('Iteraciones')
    plt.ylabel('Costo')
    plt.title('Curva de Aprendizaje del Modelo de Regresión')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calcular el costo en el conjunto de prueba para evaluar el modelo
    costo_prueba = calcular_costo(X_prueba_escalado, Y_prueba, theta)
    print("Costo en el conjunto de prueba:", costo_prueba)

    # Calcular el MSE en el conjunto de prueba como una métrica de rendimiento
    mse_prueba = np.mean((calcular_hipotesis(X_prueba_escalado, theta) - Y_prueba) ** 2)
    print("MSE en el conjunto de prueba:", mse_prueba)

    # Imprimir los parámetros theta finales
    print("Parámetros theta finales:", theta)



def main():
    interact(entrenar_y_evaluar_modelo,
             tasa_aprendizaje=FloatSlider(value=0.01, min=0.000001, max=0.1, step=0.000001, description='Tasa de Aprendizaje:', readout_format='.6f'),
             iteraciones=IntSlider(value=1000, min=100, max=10000, step=100, description='Iteraciones:'))


if __name__ == "__main__":
    main()
