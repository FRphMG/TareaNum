# Importa la biblioteca NumPy, que es fundamental para la computación científica en Python
import numpy as np

# Importa el módulo 'linalg' de NumPy, que proporciona funciones para álgebra lineal
# Se le asigna el alias 'LA' para facilitar su uso
from numpy import linalg as LA

#18 b)

def FactLU(A):
    #Realiza la factorización LU de una matriz A
    #Retorna las matrices L y U tales que A = L@U
    U = np.copy(A)
    n = A.shape[1]
    L = np.eye(n)

    for j in range(n):
        Lj = np.eye(n)
        for i in range(j+1, n):
            Lj[i,j] = -U[i,j]/U[j,j]
        L = L @ Lj
        U = Lj @ U
    L = 2*np.eye(n) - L
    return L, U

def SustDelante(L, b):
    #Realiza la sustitución hacia adelante Ly = b
    #L: matriz triangular inferior
    #b: vector de términos independientes
    #Retorna y
    x = np.zeros_like(b, dtype=float)
    n = L.shape[0]

    for i in range(n):
        sum = 0.0
        for j in range(i):
            sum += L[i,j] * x[j]
        x[i] = (b[i] - sum) / L[i,i]
    return x

def SustAtras(U, y):
    #Realiza la sustitución hacia atrás Ux = y
    #U: matriz triangular superior
    #y: vector obtenido de la sustitución hacia adelante
    #Retorna x
    x = np.zeros_like(y, dtype=float)
    n = U.shape[0]
    x[n-1] = y[n-1] / U[n-1,n-1]

    for i in range(n-2, -1, -1):
        sum = 0.0
        for j in range(i+1, n):
            sum += U[i,j] * x[j]
        x[i] = (y[i] - sum) / U[i,i]
    return x

def SolverLU(A, b):
    #Resuelve el sistema Ax = b usando factorización LU
    #A: matriz de coeficientes
    #b: vector de términos independientes
    #Retorna x: vector solución
    L, U = FactLU(A)
    y = SustDelante(L, b)
    x = SustAtras(U, y)
    return x

# Ejemplo
A = np.array([
    [2, 4, -2],
    [4, 9, -3],
    [-2, -1, 7]
])

b = np.array([2, 8, 10])

# Resolvemos el sistema
x = SolverLU(A, b)

# Mostramos los resultados paso a paso
print("Resolución del sistema Ax = b")
print("-" * 40)

print("\nMatriz A:")
print(A)
print("\nVector b:")
print(b)

L, U = FactLU(A)
print("\nFactorización LU")
print("Matriz L:")
print(np.round(L, 3))
print("\nMatriz U:")
print(np.round(U, 3))

y = SustDelante(L, b)
print("\nSolución de Ly = b:")
print("y =", np.round(y, 3))

x = SustAtras(U, y)
print("\nSolución de Ux = y:")
print("x =", np.round(x, 3))

# Verificación
print("\nVerificación:")
print("Ax =", np.round(A @ x, 3))
print("b  =", b)
print("\nError máximo:", np.max(np.abs(A @ x - b)))

