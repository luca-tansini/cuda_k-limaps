#Calcolo la pseudoinversa della trasposta e poi la traspongo
import numpy as np
A = np.random.rand(20,10)
Adag = np.linalg.pinv(A)
ATdag = np.linalg.pinv(A.transpose())
ATdagT = ATdag.transpose()
print(np.allclose(Adag,ATdagT))
