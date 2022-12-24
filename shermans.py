import numpy as np

def sherman_factorization_iterative(A):
    m, n = A.shape
    L = np.eye(n)
    U = np.zeros((n, n))

    U[0,0] = A[0,0]
    for i in range(2, n):
        a = A[:i, i]
        b = A[i, :i].T
        gamma = A[i, i]
        u = np.linalg.lstsq(L[:i, :i], a)[0]
        ell = np.linalg.lstsq(U[:i, :i].T, b)[0]
        upsilon = gamma - ell.T @ u

        U[:i, i] = u
        L[i, :i] = ell
        U[i, i] = upsilon

    return L, U

def shermans(A, i=0, L=None, U=None):
    m, n = A.shape

    if L is None:
        L = np.eye(n)
        U = np.zeros((n, n))

    if i < n:
        a = A[:i, i]
        b = A[i, :i].T

        gamma = A[i, i]
        u = np.linalg.lstsq(L[:i, :i], a)[0]
        ell = np.linalg.lstsq(U[:i, :i].T, b)[0]
        upsilon = gamma - ell.T @ u

        U[:i, i] = u
        L[i, :i] = ell
        U[i, i] = upsilon

        return sherman_factorization_recursive(A, i+1, L, U)
    else:
        return L, U

