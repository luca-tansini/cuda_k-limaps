import numpy as np

def printColumnMajorMatrix(A):
    T = A.transpose()
    for i in T:
        for j in i:
            print("{0:.5f}".format(j), end = ' ')
    print()

def printVector(v):
    for i in v:
        print("{0:.5f}".format(i), end = ' ')
    print()

#Function implementing the F-lambda shrinkage
def F(x,lambdak):
    return x * (1-np.exp(-lambdak*abs(x)))

#Function implementing k-LiMapS algorithm
def SBSk_LiMapS(k, theta, thetaPseudoInv, b, maxIter):

    #calculate initial alpha = thetaPseudoInv * b
    alpha = thetaPseudoInv @ b

    #algorithm internal loop
    i = 0;
    while(i < maxIter):
        print("\nIter #",i)

        #1b. sort sigma in descending order
        oldalpha = alpha
        sigma = sorted(np.abs(alpha))[::-1]
        print("sigma:")
        printVector(sigma)

        #2. calculate lambda = 1/sigma[k]
        lambdak = 1/sigma[k];

        #3. calculate beta = F(lambda, alpha)
        beta = F(alpha,lambdak)
        print("beta:")
        printVector(beta)

        #4. update alpha = beta - thetaPseudoInv * (theta * beta - b)
        alpha = beta - thetaPseudoInv @ (theta @ beta - b)

        #loop conditions update
        i+=1;
        norm = np.linalg.norm(alpha-oldalpha)
        print("norm:")
        print(norm,end='')
        input()
        if(norm < 1e-6):
            #print(i)
            break

    #final thresholding step: alpha[i] = 0 if |alpha[i]| <= sigma[k]
    for i in range(len(alpha)):
        if abs(alpha[i]) <= sigma[k]:
            alpha[i] = 0

    return alpha

def main():
    f = open("../input")
    n = int(f.readline())
    k = int(f.readline())
    m = n*k;

    #Read dictionary theta (read by columns [m x n], so we transpose)
    theta = []
    for i in range(m):
        tmp = []
        for j in range(n):
            tmp += [float(f.readline())]
        theta += [tmp]
    theta = np.array(theta)
    theta = theta.transpose()

    #Read theta Moore-Penrose pseudoinverse (read by columns [n x m], so we transpose)
    thetaPseudoInv = []
    for i in range(n):
        tmp = []
        for j in range(m):
            tmp += [float(f.readline())]
        thetaPseudoInv += [tmp]
    thetaPseudoInv = np.array(thetaPseudoInv)
    thetaPseudoInv = thetaPseudoInv.transpose()


    #Read optimal solution alpha
    alpha = []
    for i in range(m):
        alpha += [float(f.readline())]
    alpha = np.array(alpha)

    #Read signal b = theta * alpha (noiseless test)
    b = []
    for i in range(n):
        b += [float(f.readline())]
    b = np.array(b)

    #call SBSk_LiMapS
    SBSk_LiMapS(k, theta, thetaPseudoInv, b, 1000)

if __name__ == '__main__':
    main()
