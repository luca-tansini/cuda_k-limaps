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

#rebuild1
S = np.zeros((n,m))
S[:min(n,m), :min(n,m)] = np.diag(s)
rebuild1 = U * S * V_T

#rebuild2
S1 = np.zeros((n,m))
S1[:min(n,m), :min(n,m)] = np.diag(s1)
rebuild2 = V_T1.transpose() * S1 * U1.transpose()

print("U * S * V_T == V_T1^T * S1 * U1^T == A?: ",end='')
print(np.allclose(rebuild1, rebuild2) and np.allclose(rebuild2, A))

print("s:")
print(s.shape)
print(s)

SPseudoInv = S.copy()
for i in range(min(n,m)):
    if SPseudoInv[i,i] != 0:
        SPseudoInv[i,i] = 1 / SPseudoInv[i,i]
SPseudoInv = SPseudoInv.transpose()

print("APseudoInv = V_T^T * SPseudoInv * U^T")
APseudoInv  = V_T.transpose() * SPseudoInv * U.transpose()
print("APseudoInv1 = U1^T * SPseudoInv * V_T1")
APseudoInv1 = U1.transpose() * SPseudoInv * V_T1

print("APseudoInv == APseudoInv1 ?: ", np.allclose(APseudoInv, APseudoInv1))

print("A*APseudoInv*A == A?: ", end='')
res = A*APseudoInv*A
print(np.allclose(res,A))
print("APseudoInv*A*APseudoInv == APseudoInv?: ", end='')
res = APseudoInv*A*APseudoInv
print(np.allclose(res,APseudoInv))

print("A*APseudoInv1*A == A?: ", end='')
res = A*APseudoInv1*A
print(np.allclose(res,A))
print("APseudoInv1*A*APseudoInv1 == APseudoInv1?: ", end='')
res = APseudoInv1*A*APseudoInv1
print(np.allclose(res,APseudoInv1))
