import sys
import numpy as np
import K_LiMapS

def printColumnMajorMatrix(A):
    T = A.transpose()
    for i in T:
        for j in i:
            print("{0:.7f}".format(j))

def printVector(v):
    for i in v:
        print("{0:.7f}".format(i))

def main():
    n = int(sys.argv[1])
    k = int(sys.argv[2])
    print(n)
    print(k)

    #Randomly generate dictionary
    m = n*k
    theta = np.random.rand(n,m)
    printColumnMajorMatrix(theta)

    #Calculate dictionary pseudoinv
    thetaPseudoInv = np.linalg.pinv(theta)
    printColumnMajorMatrix(thetaPseudoInv)

    #Randomly generate optimal solution alpha
    values = np.random.rand(k)
    alpha = np.append(values,np.zeros(m-k))
    np.random.shuffle(alpha)
    printVector(alpha)

    #Calculate b = theta * alpha
    b = theta @ alpha
    printVector(b)

    #Print initial alpha
    printVector(thetaPseudoInv @ b)

    #Print limaps solution
    limapsSolution = K_LiMapS.k_LiMapS(k, theta, thetaPseudoInv, b, 1000)
    printVector(limapsSolution)

if __name__ == '__main__':
    main()
