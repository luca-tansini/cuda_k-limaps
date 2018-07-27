import numpy as np
import math
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
    theta = np.random.rand(n,m)

    #Calculate dictionary pseudoinv
    thetaPseudoInv = MoorePenrosePseudoInverse(theta,n,m)

    #Randomly generate optimal solution alpha
    values = np.random.rand(k)
    alpha = np.append(values,np.zeros(m-k))
    np.random.shuffle(alpha)
    #print("\nalpha:")
    #print(alpha)

    #Calculate b = theta * alpha
    b = theta @ alpha

    #Call k_LiMapS
    limapsSolution = k_LiMapS(k, theta, thetaPseudoInv, b, maxIter)
    return not max(abs(alpha - limapsSolution)) > 0.001

def runTest():
    nsizes = [10,25,50,100]
    results = []

    for i in nsizes:
        for k in range(2,i//4+1,math.ceil((i//4+1)/10)):
            success = 0
            for j in range(100):
                if(k_LiMapS_noiselessTest(i,k,1000)): success += 1
            print((i,k,success))
            results += [(i,k,success)]

    for r in results:
        print(r)
