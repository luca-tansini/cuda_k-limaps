import numpy as np
import math
from K_LiMapS import *
import time
import sys

def k_LiMapS_noiselessTest(n,iters=50):

    print("    n|     m| delta|     k|   rho|  succ%|      avgMSE      | avgTime |");
    for m in range(n,10*n+1,n):

        #Randomly generate dictionary and normalize columns
        D = np.random.randn(n,m)
        for i in range(m):
            D[:,i] /= np.linalg.norm(D[:,i])

        #Calculate dictionary pseudoinv
        DINV = np.linalg.pinv(D)

        for k in range(math.ceil(n/10),n//2+1,math.ceil(n/20)):

            print("{0:5d}| {1:5d}| {2:5.2f}| {3:5d}| {4:5.2f}| ".format(n,m,n/m,k,k/n), end='')

            avgMSE = 0
            avgTime = 0
            succ = 0
            for i in range(iters):

                #Randomly generate optimal solution alpha
                values = np.random.randn(k)
                alphaopt = np.append(values,np.zeros(m-k))
                np.random.shuffle(alphaopt)

                #Calculate s = D * alphaopt
                s = D @ alphaopt

                #Call k_LiMapS
                t = time.time()
                alphalimaps = k_LiMapS(k, D, DINV, s, 1000)
                avgTime += time.time() - t

                #Calculate MSE
                diff = s - D @ alphalimaps
                MSE = sum(diff**2)/n
                avgMSE += MSE

                #check succ
                for j in range(m):
                    if(abs(alphaopt[j] - alphalimaps[j]) > 1e-3):
                        break
                if(j == m-1):
                    succ += 1

            avgMSE  /= iters
            avgTime /= iters
            print("{0:6.2f}| {1:17.15f}| {2:8.6f}|".format(succ*100/5, avgMSE, avgTime))

    return

if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print("usage: noiselessTest <n> [<iters>]")
        sys.exit(-1)
    n = int(sys.argv[1])
    k_LiMapS_noiselessTest(n)
