# Importar la biblioteca NumPy y asignarle el alias 'np'
import numpy as np

# Importar el módulo pyplot de Matplotlib y asignarle el alias 'plt'
import matplotlib.pyplot as plt

A = LL^T


def Cholesky(A):
    # Obtener el tamaño de la matriz A
    n = len(A)
    # Inicializar la matriz L con ceros, del mismo tamaño que A
    L = np.zeros_like(A)

    # Iterar sobre cada fila de la matriz
    for i in range(n):
        # Iterar sobre cada columna de la matriz hasta la diagonal
        for j in range(i + 1):
            if i == j:
                # Si estamos en la diagonal, calcular la raíz cuadrada
                sum = 0.0
                for k in range(j):
                    sum += L[j][k] * L[j][k]
                L[j][j] = np.sqrt(A[j][j] - sum)
            else:
                # Si no estamos en la diagonal, calcular el valor de L[i][j]
                sum = 0.0
                for k in range(j):
                    sum += L[i][k] * L[j][k]
                L[i][j] = (A[i][j] - sum) / L[j][j]

    # Retornar la matriz triangular inferior L
    return L


A = LDL^T



import numpy as np

def Diagonal(A):
    # Obtener el tamaño de la matriz A
    n = len(A)

    # Crear matrices D y L de ceros
    D = np.zeros((n, n))  # Matriz diagonal
    L = np.zeros((n, n))  # Matriz triangular inferior

    # Iterar sobre cada columna de la matriz
    for j in range(n):
        sum = 0.0
        # Calcular la suma para la diagonal de D
        for k in range(j):
            sum += L[j][k] * D[k][k] * L[j][k]
        D[j][j] = np.sqrt(A[j][j] - sum)  # Calcular el valor de D[j][j]

        # Calcular los elementos no diagonales de L
        for i in range(j + 1, n):
            sum = 0.0
            for k in range(j):
                sum += L[i][k] * D[k][k] * L[j][k]
            L[i][j] = (A[i][j] - sum) / D[j][j]  # Calcular L[i][j]

    # Retornar las matrices L y D
    return L, D

