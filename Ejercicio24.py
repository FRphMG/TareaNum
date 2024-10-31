# Importar la biblioteca NumPy y asignarle el alias 'np'
import numpy as np

# Importar el módulo pyplot de Matplotlib y asignarle el alias 'plt'
import matplotlib.pyplot as plt

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


A = LL^T



#Inciso A
# Definir la matriz A para la que se desea calcular la descomposición de Cholesky
A=np.array([[2.,-1.,0.],[-1.,2.,-1.],[0.,-1.,2.]])

# Calcular la matriz L usando la función Cholesky
L = Cholesky(A)  # L ya está calculada

# Imprimir el producto de L por su transpuesta
print(L@L.T) # Utiliza la operación @ para multiplicar matrices

#Inciso B
# Definir la matriz B para la que se desea calcular la descomposición de Cholesky
A=np.array([[4.,1.,1.,1.],[1.,3.,-1.,1],[1.,-1.,2.,0.],[1.,1.,0.,2.]])

# Calcular la matriz L usando la función Cholesky
L = Cholesky(A)  # L ya está calculada

# Imprimir el producto de L por su transpuesta
print(L@L.T) # Utiliza la operación @ para multiplicar matrices

#Inciso C
# Definir la matriz C para la que se desea calcular la descomposición de Cholesky
A=np.array([[4.,1.,-1.,0.],[1.,3.,-1.,0.],[-1.,-1.,5.,2.],[0.,0.,2.,4.]])

# Calcular la matriz L usando la función Cholesky
L = Cholesky(A)  # L ya está calculada

# Imprimir el producto de L por su transpuesta
print(L@L.T) # Utiliza la operación @ para multiplicar matrices

#Inciso D
# Definir la matriz D para la que se desea calcular la descomposición de Cholesky
A=np.array([[6.,2.,1.,-1.],[2.,4.,1.,0.],[1.,1.,4.,-1.],[-1.,0.,-1.,3.]])

# Calcular la matriz L usando la función Cholesky
L = Cholesky(A)  # L ya está calculada

# Imprimir el producto de L por su transpuesta
print(L@L.T) # Utiliza la operación @ para multiplicar matrices

import numpy as np

def Diagonal(A):
    n = len(A)
    D = np.zeros((n,n))  # Crear matriz D de ceros
    L = np.zeros((n,n))  # Crear matriz L de ceros

    for j in range(n):
        sum = 0.0
        for k in range(j):
            sum+= L[j][k] * D[k][k] * L[j][k]
        D[j][j] = np.sqrt(A[j][j] - sum)

        for i in range(j + 1, n):
            sum = 0.0
            for k in range(j):
                sum += L[i][k] * D[k][k] * L[j][k]
            L[i][j] = (A[i][j] - sum) / D[j][j]

    return L, D

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



A = LDL^T

#Inciso A
# Definir la matriz A
A=np.array([[2.,-1.,0.],[-1.,2.,-1.],[0.,-1.,2.]])

L, D = Diagonal(A)  # Calcular L y D usando la función Diagonal

# Imprimir el producto de L, D y la transpuesta de L
print(L @ D @ L.T)  # Utiliza la operación @ para multiplicar matrices

#Inciso B
# Definir la matriz B
A=np.array([[4.,1.,1.,1.],[1.,3.,-1.,1],[1.,-1.,2.,0.],[1.,1.,0.,2.]])

L, D = Diagonal(A)  # Calcular L y D usando la función Diagonal

# Imprimir el producto de L, D y la transpuesta de L
print(L @ D @ L.T)  # Utiliza la operación @ para multiplicar matrices

#Inciso C
# Definir la matriz C
A=np.array([[4.,1.,-1.,0.],[1.,3.,-1.,0.],[-1.,-1.,5.,2.],[0.,0.,2.,4.]])

L, D = Diagonal(A)  # Calcular L y D usando la función Diagonal

# Imprimir el producto de L, D y la transpuesta de L
print(L @ D @ L.T)  # Utiliza la operación @ para multiplicar matrices

#Inciso D
# Definir la matriz D
A=np.array([[6.,2.,1.,-1.],[2.,4.,1.,0.],[1.,1.,4.,-1.],[-1.,0.,-1.,3.]])

L, D = Diagonal(A)  # Calcular L y D usando la función Diagonal

# Imprimir el producto de L, D y la transpuesta de L
print(L @ D @ L.T)  # Utiliza la operación @ para multiplicar matrices
