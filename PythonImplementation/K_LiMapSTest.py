import numpy as np
import math
from K_LiMapS import *

#Test with b obtained as b = theta * alpha
def k_LiMapS_noiselessTest(n,maxIter):

    res = []
    for m in range(n,5*n+1,n):
        print("\nm={0} , delta={1:.2f}:".format(m,n/m))
        #Randomly generate dictionary
        theta = np.random.rand(n,m)

        #Calculate dictionary pseudoinv
        thetaPseudoInv = MoorePenrosePseudoInverse(theta,n,m)

        mres = []
        for k in range(math.ceil(n/10),n//2+1,math.ceil(n/10)):
            meanMSE = 0
            for i in range(100):

                #Randomly generate optimal solution alpha
                values = np.random.rand(k)
                alpha = np.append(values,np.zeros(m-k))
                np.random.shuffle(alpha)

                #Calculate b = theta * alpha
                b = theta @ alpha

                #Call k_LiMapS
                limapsSolution = k_LiMapS(k, theta, thetaPseudoInv, b, maxIter)

                #Calculate MSE
                diff = b - theta @ limapsSolution
                MSE = sum(diff**2)/n
                meanMSE += MSE

            meanMSE /= 100
            print("    k={0} , rho={1} --> MSE medio: {2}".format(k,k/n,meanMSE))
            mres += [(k,meanMSE)]
        res += [(m,mres)]
    return
