import numpy as np
import sys

n = int(sys.argv[1])
k = int(sys.argv[2])
m = n*k

A = np.random.rand(n,m)
A = np.matrix(A)
print("A: matrix [{0}x{1}]".format(n,m))
T = A.transpose()
print("T = A^T (A transpose)")
U, s, V = np.linalg.svd(A, full_matrices=True)
print("SVD(A) = U * S * V")
Ut, st, Vt = np.linalg.svd(T, full_matrices=True)
print("SVD(T) = Ut * St * Vt")

print("U == Vt ?: ",end='')
print(np.allclose(U,Vt.transpose()), end = ', ')
print("in absolute values: ",end='')
print(np.allclose(np.absolute(U),np.absolute(Vt.transpose())))

print("s == st ?: ",end='')
print(np.allclose(s,st))

print("V == Ut ?: ",end='')
print(np.allclose(V,Ut.transpose()), end = ', ')
print("in absolute values: ",end='')
print(np.allclose(np.absolute(V),np.absolute(Ut.transpose())))

#rebuild1
S = np.zeros((n,m))
S[:min(n,m), :min(n,m)] = np.diag(s)
rebuild1 = U * S * V

#rebuild2
St = np.zeros((n,m))
St[:min(n,m), :min(n,m)] = np.diag(st)
rebuild2 = Vt.transpose() * St * Ut.transpose()

print("U * S * V == Vt^T * St * Ut^T ?: ",end='')
print(np.allclose(rebuild1, rebuild2))

print("s:")
print(s.shape)
print(s)
