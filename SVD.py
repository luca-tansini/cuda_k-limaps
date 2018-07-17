import numpy as np

a = [[1,2,3,4,5,6],[7,8,9,0,1,2]]
a = np.matrix(a)
print(a)
U, s, V = np.linalg.svd(a, full_matrices=True)
#V is actually VH and s is only a vector, we have to create the diag matrix
S = np.zeros((2,6))
S[:2, :2] = np.diag(s)
print(U)
print(S)
print(V)
print(U*S*V)
