# Yldefonso Hérnandez Danna Paola
# Guzmán Martínez Frida Paola

# Importa la biblioteca NumPy, que es fundamental para la computación científica en Python
import numpy as np

# Importa el módulo 'linalg' de NumPy, que proporciona funciones para álgebra lineal
# Se le asigna el alias 'LA' para facilitar su uso
from numpy import linalg as LA
#Ejercicio 19

def FactLU(A):
    # Copia la matriz A en U, que se modificará durante el proceso
    U = np.copy(A)

    # Obtiene el número de columnas (o filas, suponiendo que A es cuadrada) de A
    n = A.shape[1]

    # Inicializa la matriz L como la matriz identidad de tamaño n
    L = np.eye(n)

    # Recorre cada columna de la matriz
    for j in range(n):
        # Crea una matriz identidad que se usará para modificar L y U
        Lj = np.eye(n)

        # Para cada fila debajo de la fila actual (j), calcula los coeficientes para L
        for i in range(j + 1, n):
            # Calcula el factor para la matriz L
            Lj[i, j] = -U[i, j] / U[j, j]

        # Actualiza L multiplicándola por la matriz Lj
        L = L @ Lj

        # Actualiza U multiplicándola por la matriz Lj
        U = Lj @ U

    # Ajusta L para que contenga la forma correcta de la matriz triangular inferior
    L = 2 * np.eye(n) - L

    # Retorna las matrices L y U
    return L, U

FACTORIZACIÓN LU

#---Factorizacion LU.
#Inciso A
# Define una matriz A
A=np.array([[4.,-1.,3.],[-8.,4.,-7.],[12.,1.,8]])
# Llama a la función FactLU para obtener las matrices L y U
L, U = FactLU(A)
# Muestra las matrices L y U
L,U


#---Factorizacion LU.
#Inciso B
# Define una matriz A
A = np.array([[1., 4., -2., 1.], [-2., -4., -3., 1.], [1., 16., -17., 9.], [2., 4., -9., -3.]])
# Llama a la función FactLU para obtener las matrices L y U
L, U = FactLU(A)
# Muestra las matrices L y U
L,U



#---Factorizacion LU.
#Inciso C
# Define una matriz A
A=np.array([[4.,5.,-1.,2.,3],[12.,13.,0.,10.,3],[-8.,-8.,5.,-11.,4],[16.,18.,-7.,20.,4],[-4.,-9.,31.,-31.,-1]])
# Llama a la función FactLU para obtener las matrices L y U
L, U = FactLU(A)
# Muestra las matrices L y U
L,U


PIVOTEO PARCIAL

import numpy as np

def FactLU(A):
    # Crea una copia de la matriz A para trabajar con U
    U = np.copy(A)
    # Obtiene el número de columnas (o filas) de A
    n = A.shape[1]
    # Inicializa la matriz L como la matriz identidad
    L = np.eye(n)

    # Recorre cada columna de la matriz
    for j in range(n):
        Lj = np.eye(n)  # Crea una matriz identidad para modificar L y U
        # Calcula los factores para la matriz L
        for i in range(j + 1, n):
            Lj[i, j] = -U[i, j] / U[j, j]

        # Actualiza L y U
        L = L @ Lj
        U = Lj @ U

    # Ajusta L para tener la forma correcta
    L = 2 * np.eye(n) - L

    return L, U

def SustDelante(L, b):
    # Resuelve Lx = b utilizando sustitución hacia adelante
    x = np.zeros_like(b)
    n = L.shape[0]  # Número de filas de L
    for i in range(n):
        suma = 0.0
        for j in range(i):
            suma += L[i, j] * x[j]
        # Calcula la solución para la variable x[i]
        x[i] = (b[i] - suma) / L[i, i]

    return x

def SustAtras(U, y):
    # Resuelve Ux = y utilizando sustitución hacia atrás
    x = np.zeros_like(y)
    n = U.shape[0]  # Número de filas de U
    # Calcula la última variable
    x[n-1] = y[n-1] / U[n-1][n-1]

    for i in range(n-2, -1, -1):
        suma = 0.0
        for j in range(i + 1, n):
            suma += U[i, j] * x[j]
        # Calcula la solución para la variable x[i]
        x[i] = (y[i] - suma) / U[i, i]

    return x

def SolverLU(A, b):
    # Resuelve el sistema Ax = b usando factorización LU
    L, U = FactLU(A)  # Factoriza A en L y U
    y = SustDelante(L, b)  # Resuelve Ly = b
    x = SustAtras(U, y)    # Resuelve Ux = y

    return x

def Permutaciones(A, b):
    # Realiza el pivoteo parcial y devuelve la matriz U, el vector b y las matrices de permutación
    U = np.copy(A)
    x = np.copy(b)
    Ps = []  # Lista para almacenar las matrices de permutación

    for j in range(U.shape[0]):
        P = np.eye(U.shape[0])  # Matriz de permutación
        # Encuentra el índice del máximo elemento en la columna j
        k = np.argmax(np.abs(U[j:, j])) + j
        # Intercambia filas en U, P y b
        U[[j, k]] = U[[k, j]]
        P[[j, k]] = P[[k, j]]
        b[[j, k]] = b[[k, j]]

        Ps.append(P)  # Guarda la matriz de permutación

    return Ps, U, b


#---Factorizacion LU con pivoteo parcial.
# Inciso A
np.argmax(A[0,:])  # Encuentra el valor máximo de la primera fila de la matriz
A[[0, 2]] = A[[2, 0]]  # Intercambia la primera y la tercera fila de la matriz
A = np.array([[4., -1., 3.], [-8., 4., -7.], [12., 1., 8]])  # Define la matriz A
y = np.array([-8, 19, -19])  # Define el vector b
Ps, A_g, b_g = Permutaciones(A, y)  # Aplica el pivoteo parcial
x = SolverLU(A_g, b_g)  # Resuelve el sistema Ax = b utilizando LU
x  # Muestra la solución x


#---Factorizacion LU con pivoteo parcial.
#Inciso B
np.argmax(A[0,:]) # Encuentra el valor máximo de la matriz
A[[0,3]]=A[[3,0]]# Intercambia el renglon 0 con el 3
A = np.array([[1., 4., -2., 1.], [-2., -4., -3., 1.], [1., 16., -17., 9.], [2., 4., -9., -3.]]) # Define la matriz A
y=np.array([3.5,-2.5,15,10.5]) # Define el vector b
Ps,A_g,b_g=Permutaciones(A,y)  # Aplica el pivoteo parcial
x=SolverLU(A_g,b_g)  # Resuelve el sistema Ax = b utilizando LU
x # Muestra la solución x

#---Factorizacion LU con pivoteo parcial.
#Inciso C
np.argmax(A[0,:]) # Encuentra el valor máximo de la matriz
A[[0,3]]=A[[3,0]]# Intercambia el renglon 0 con el 3
A=np.array([[4.,5.,-1.,2.,3],[12.,13.,0.,10.,3],[-8.,-8.,5.,-11.,4],[16.,18.,-7.,20.,4],[-4.,-9.,31.,-31.,-1]])  # Define la matriz A
y=np.array([34,93,-33,131,-58]) # Define el vector b
Ps,A_g,b_g=Permutaciones(A,y)  # Aplica el pivoteo parcial
x=SolverLU(A_g,b_g)  # Resuelve el sistema Ax = b utilizando LU
x # Muestra la solución x
