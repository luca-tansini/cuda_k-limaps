import numpy as np
import math
import sys
import time
from K_LiMapS import *

#Test with b obtained as b = theta * alpha
def simpleTest(n,m,k,iters):

    t1 = time.time()
    #Randomly generate dictionary
    theta = np.random.rand(n,m)

    #Calculate dictionary pseudoinv
    thetaPseudoInv = MoorePenrosePseudoInverse(theta,n,m)
    print("Dictionary generation and pseudoinverse computation time elapsed: {0:.6f}".format(time.time() - t1))

    succ = 0
    avgt = 0

    for i in range(iters):

        #Randomly generate optimal solution alpha
        values = np.random.rand(k)
        alpha = np.append(values,np.zeros(m-k))
        np.random.shuffle(alpha)

        #Calculate b = theta * alpha
        b = theta @ alpha

        t1 = time.time()
        #Call k_LiMapS
        limapsSolution = k_LiMapS(k, theta, thetaPseudoInv, b, 1000)
        avgt += time.time() - t1

        #Check result
        if( max(abs(alpha - limapsSolution)) < 0.0001): succ += 1

    print("\n{0:.2f}%".format(100*succ/iters))
    print("\nAverage k-LiMapS execution time: {0:.6f}".format(avgt/iters))
    return

if __name__ == '__main__':
    n = int(sys.argv[1])
    m = int(sys.argv[2])
    k = int(sys.argv[3])
    iters = int(sys.argv[4])
    simpleTest(n,m,k,iters)
