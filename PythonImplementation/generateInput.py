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
    m = int(sys.argv[2])
    k = int(sys.argv[3])
    iters = int(sys.argv[4])
    print(n)
    print(m)
    print(k)
    print(iters)

    #Randomly generate dictionary
    theta = np.random.rand(n,m)
    printColumnMajorMatrix(theta)

    #Calculate dictionary pseudoinv
    thetaPseudoInv = np.linalg.pinv(theta)
    printColumnMajorMatrix(thetaPseudoInv)

    #Randomly generate optimal solutions alpha
    for i in range(iters):
        values = np.random.rand(k)
        alpha = np.append(values,np.zeros(m-k))
        np.random.shuffle(alpha)
        printVector(alpha)

if __name__ == '__main__':
    main()
