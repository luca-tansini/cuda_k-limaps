import numpy as np

a = [[0.782,0.195,0.264],[0.902,0.477,0.718],[0.491,0.593,0.069],[0.037,0.932,0.421],[0.977,0.716,0.709],[0.541,0.021,0.934],[0.739,0.585,0.904]]
a = np.matrix(a)
print(a,end="\n\n")
U, s, V = np.linalg.svd(a, full_matrices=True)
#V is actually VH and s is only a vector, we have to create the diag matrix
S = np.zeros((7,3))
S[:3, :3] = np.diag(s)
print(U,end="\n\n")
print(S,end="\n\n")
print(V,end="\n\n")
print(U*S*V,end="\n\n")
