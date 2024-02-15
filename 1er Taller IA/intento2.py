import numpy as np


# 1. Función para calcular la hipótesis h_𝜃(X)
def calculate_hypothesis(X, theta):
    """
    Calcula la hipótesis h_𝜃(X) para un conjunto de datos X y parámetros 𝜃.

    Parámetros:
    - X: Matriz de entrada (m x n), donde m es el número de ejemplos y n el número de características.
    - theta: Vector de parámetros (n+1 x 1), incluyendo el término de sesgo (bias).

    Retorna:
    - La hipótesis calculada para cada ejemplo de entrada.
    """
    return np.dot(X, theta)


# 2. Función para calcular el valor de la función de costo J(𝜃)
def compute_cost(X, y, theta):
    """
    Calcula el valor de la función de costo J(𝜃) para una hipótesis dada y las etiquetas reales.

    Parámetros:
    - X: Matriz de entrada.
    - y: Vector de etiquetas reales.
    - theta: Vector de parámetros.

    Retorna:
    - El valor de la función de costo.
    """
    m = len(y)  # Número de ejemplos de entrenamiento
    h = calculate_hypothesis(X, theta)
    return (1 / (2 * m)) * np.sum((h - y) ** 2)


# 3. Función para calcular el gradiente de la función de costo
def compute_gradient(X, y, theta):
    """
    Calcula el gradiente de la función de costo respecto a los parámetros 𝜃.

    Parámetros:
    - X: Matriz de entrada.
    - y: Vector de etiquetas reales.
    - theta: Vector de parámetros.

    Retorna:
    - El gradiente de la función de costo.
    """
    m = len(y)
    h = calculate_hypothesis(X, theta)
    return (1 / m) * np.dot(X.T, (h - y))


# 4. Función para actualizar los parámetros 𝜃
def update_parameters(theta, learning_rate, gradient):
    """
    Actualiza los parámetros 𝜃 utilizando el gradiente de la función de costo.

    Parámetros:
    - theta: Vector de parámetros actual.
    - learning_rate: Tasa de aprendizaje.
    - gradient: Gradiente de la función de costo.

    Retorna:
    - Los parámetros 𝜃 actualizados.
    """
    return theta - learning_rate * gradient

# Incorporar estas funciones en el proceso de entrenamiento y visualización puede seguir un enfoque similar
# al mostrado anteriormente, donde se llaman estas funciones en lugar de tener el código directamente
# en el bucle del gradiente descendiente o en la función de visualización.
