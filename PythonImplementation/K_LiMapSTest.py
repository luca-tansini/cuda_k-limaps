import numpy as np
from K_LiMapS import *

#Test for MoorePenrosePseudoInverse
def MoorePenrosePseudoInverseTest(n,m):
    A = np.random.rand(n,m)
    APseudoInv = MoorePenrosePseudoInverse(A,n,m)
    Apinv = np.linalg.pinv(A)

    print("APseudoInv == Apinv ? ", np.allclose(APseudoInv,Apinv))
    print("A*APseudoInv*A == A?: ", end='')
    res = A*APseudoInv*A
    print(np.allclose(res,A))
    print("APseudoInv*A*APseudoInv == APseudoInv?: ", end='')
    res = APseudoInv*A*APseudoInv
    print(np.allclose(res,APseudoInv))

#Test with b obtained as b = theta * alpha
def k_LiMapS_noiselessTest(n,k,maxIter):

    #Randomly generate dictionary
    m = n*k
    theta = np.matrix(np.random.rand(n,m))

    #Calculate dictionary pseudoinv
    thetaPseudoInv = MoorePenrosePseudoInverse(theta,n,m)

    #Randomly generate optimal solution alpha
    values = np.random.rand(k)
    alpha = np.append(values,np.zeros(m-k))
    np.random.shuffle(alpha)
    alpha = np.matrix(alpha).transpose()
    print("\nalpha:")
    print(alpha)

    #Calculate b = theta * alpha
    b = theta * alpha

    #Call k_LiMapS
    limapsSolution = k_LiMapS(k, theta, thetaPseudoInv, b, maxIter)
    print("\nlimapsSolution:")
    print(limapsSolution)

    print("diff:")
    print(alpha - limapsSolution)

    print(not max(abs(alpha - limapsSolution)) > 0.001)
