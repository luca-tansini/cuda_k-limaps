import numpy as np
import math
import sys
import time
from K_LiMapS import *

def simpleTest(inputfile):

    f = open(inputfile)
    n = int(f.readline())
    m = int(f.readline())
    k = int(f.readline())
    iters = int(f.readline())

    #Read dictionary theta (read by columns [m x n], so we transpose)
    theta = []
    for i in range(m):
        tmp = []
        for j in range(n):
            tmp += [float(f.readline())]
        theta += [tmp]
    theta = np.array(theta)
    theta = theta.transpose()

    #Read dictionary pseudoinverse (read by columns [n x m], so we transpose)
    thetaPseudoInv = []
    for i in range(n):
        tmp = []
        for j in range(m):
            tmp += [float(f.readline())]
        thetaPseudoInv += [tmp]
    thetaPseudoInv = np.array(thetaPseudoInv)
    thetaPseudoInv = thetaPseudoInv.transpose()

    avgMSE = 0
    avgt = 0
    succ = 0

    for iter in range(iters):

        #Read optimal solution alpha
        alpha = []
        for i in range(m):
            alpha += [float(f.readline())]
        alpha = np.array(alpha)

        #Calculate b = theta * alpha
        b = theta @ alpha

        t1 = time.time()
        #Call k_LiMapS
        limapsSolution = k_LiMapS(k, theta, thetaPseudoInv, b, 1000)
        avgt += time.time() - t1

        #Check result
        if( max(abs(alpha - limapsSolution)) < 0.0001): succ += 1
        diff = b - theta @ limapsSolution
        MSE = sum(diff**2)/n
        avgMSE += MSE

    avgMSE /= iters
    print("\nSucces percentage: {0:.2f}%".format(100*succ/iters))
    print("\nAverage MSE: {0:.15f}".format(avgMSE))
    print("\nAverage k-LiMapS execution time: {0:.6f}".format(avgt/iters))
    return

if __name__ == '__main__':
    inputfile = sys.argv[1]
    simpleTest(inputfile)
