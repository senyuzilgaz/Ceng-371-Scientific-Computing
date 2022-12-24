import numpy as np

def iterative_crout(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        U[i, i] = 1
    for j in range(n):
        for i in range(j, n):
            L[i, j] = A[i, j] - (L[i, 0:j] @ U[0:j, j])
        for i in range(j, n):
            U[j, i] = (A[j, i] - (L[j, 0:j] @ U[0:j, i])) / L[j, j]
    return L, U


def compute_lu(A, L, U, j, n):
    for i in range(j, n):
        L[i, j] = A[i, j] - (L[i, 0:j] @ U[0:j, j])
    for i in range(j, n):
        U[j, i] = (A[j, i] - (L[j, 0:j] @ U[0:j, i])) / L[j, j]
    if j < n - 1:
        compute_lu(A, L, U, j + 1, n)

def crouts(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        U[i, i] = 1
    compute_lu(A, L, U, 0, n)
    return L, U
