# Importaciones necesarias de librerías
import numpy as np  # Para operaciones numéricas y manejo de arrays
import pandas as pd  # Para la manipulación y análisis de datos
from sklearn.model_selection import train_test_split  # Para dividir los datos en conjuntos de entrenamiento y prueba
from sklearn.preprocessing import StandardScaler  # Para la normalización de los datos
import matplotlib.pyplot as plt  # Para la visualización de datos
from ipywidgets import interact, FloatSlider  # Para crear interfaces interactivas en Jupyter notebooks
#%matplotlib inline

# Cargar los datos desde un archivo Excel
data = pd.read_excel('Real estate valuation data set.xlsx')

# Preparación de los datos:
# Selecciona las características relevantes (X) y la variable objetivo (Y) del dataset
X = data[['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']].values
Y = data['Y house price of unit area'].values.reshape(-1, 1)

# División de los datos en conjuntos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalización de los datos para mejorar el rendimiento del algoritmo de gradiente descendiente
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)  # Ajusta y transforma los datos de entrenamiento
X_test_scaled = scaler_X.transform(X_test)  # Transforma los datos de prueba
# Añade una columna de unos a X para representar el término de intercepción (bias)
X_train_scaled = np.hstack([np.ones((X_train_scaled.shape[0], 1)), X_train_scaled])
X_test_scaled = np.hstack([np.ones((X_test_scaled.shape[0], 1)), X_test_scaled])

# Definición de la función de costo
def compute_cost(X, y, theta):
    m = len(y)  # Número de ejemplos de entrenamiento
    J = (1/(2*m)) * np.sum((X.dot(theta) - y) ** 2)  # Cálculo del costo
    return J

# Implementación del algoritmo de gradiente descendiente
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = []  # Para almacenar el historial de costo
    for i in range(iterations):
        # Actualización de theta según la regla del gradiente descendiente
        theta = theta - (alpha/m) * (X.T.dot(X.dot(theta) - y))
        J_history.append(compute_cost(X, y, theta))  # Guarda el costo en cada iteración
    return theta, J_history

# Función para la visualización interactiva de la curva de aprendizaje
def plot_learning_curve(alpha):
    theta_initial = np.zeros((X_train_scaled.shape[1], 1))  # Inicialización de theta con ceros
    iterations = 100  # Número de iteraciones
    _, J_history = gradient_descent(X_train_scaled, Y_train, theta_initial, alpha, iterations)
    plt.figure(figsize=(10, 6))
    plt.plot(J_history, '-b')  # Grafica la curva de aprendizaje
    plt.xlabel('Number of iterations')  # Etiqueta del eje X
    plt.ylabel('Cost J')  # Etiqueta del eje Y
    plt.title('Learning Curve for alpha = {}'.format(alpha))  # Título de la gráfica
    plt.show()

# Creación de un widget interactivo para ajustar la tasa de aprendizaje (alpha) y visualizar los efectos
interact(plot_learning_curve, alpha=FloatSlider(value=0.01, min=0.001, max=0.1, step=0.001, description='Learning Rate:', readout_format='.3f'));
