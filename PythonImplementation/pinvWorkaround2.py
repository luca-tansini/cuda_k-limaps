#Genero randomicamente una m*n che sarà la pseudoinversa e poi calcolo A applicando la pseudoinversa. (meno operazioni ma la distribuzione randomica viene applicata alla pinv e non ad A)
import numpy as np
Apinv = np.random.rand(10,20)
A = np.linalg.pinv(Apinv)
Apinv2 = np.linalg.pinv(A)
print(np.allclose(Apinv,Apinv2))
