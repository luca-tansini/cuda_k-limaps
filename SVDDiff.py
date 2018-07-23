import numpy as np
import sys

n = int(sys.argv[1])
k = int(sys.argv[2])
m = n*k

A = np.random.rand(n,m)
A = np.matrix(A)
print("A: matrix [{0}x{1}]".format(n,m))
U, s, V_T = np.linalg.svd(A, full_matrices=True)
print("SVD(A) = U * S * V_T")
U1, s1, V_T1 = np.linalg.svd(A.transpose(), full_matrices=True)
print("SVD(A^T) = U1 * S1 * V_T1")

print("U == V_T1^T ?: ",end='')
print(np.allclose(U,V_T1.transpose()), end = ', ')
print("in absolute values: ",end='')
print(np.allclose(np.absolute(U),np.absolute(V_T1.transpose())))

print("s == s1 ?: ",end='')
print(np.allclose(s,s1))

print("V_T == U1^T ?: ",end='')
print(np.allclose(V_T,U1.transpose()), end = ', ')
print("in absolute values: ",end='')
print(np.allclose(np.absolute(V_T),np.absolute(U1.transpose())))

print("Positions of sign differences between U and V_T1^T:")
V1 = V_T1.transpose()
for i in range(n):
    for j in range(n):
        if(U[i,j] == -1*V1[i,j]): print((i,j))

print("\nPositions of sign differences between V_T and U1^T:")
U1_T = U1.transpose()
for i in range(m):
    for j in range(m):
        if(V_T[i,j] == -1*U1_T[i,j]): print((i,j))
