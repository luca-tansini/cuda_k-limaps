import numpy as np
import time

#Calculates MoorePenrose pseudo-inverse of matrix A via SVD
#Being SVD(A) = U * S * V^T, A^+ = V * S^+ * U^T
def MoorePenrosePseudoInverse(A,n,m):

    #Calculate SVD(A)
    U, s, V_T = np.linalg.svd(A, full_matrices=True)

    #Calculate S^+
    S = [ 1/x if abs(x) > np.finfo(float).eps else 0 for x in s]
    SPseudoInv = np.zeros((n,m))
    SPseudoInv[:min(n,m), :min(n,m)] = np.diag(S)
    SPseudoInv = SPseudoInv.transpose()

    return V_T.transpose() @ SPseudoInv @ U.transpose()

#Function implementing the F-lambda shrinkage
def F(x,lambdak):
    return x * (1-np.exp(-lambdak*abs(x)))

#Function implementing k-LiMapS algorithm
def k_LiMapS(k, theta, thetaPseudoInv, b, maxIter):

    #calculate initial alpha = thetaPseudoInv * b
    alpha = thetaPseudoInv @ b

    #algorithm internal loop
    i = 0;
    while(i < maxIter):

        #1b. sort sigma in descending order
        oldalpha = alpha
        sigma = sorted(np.abs(alpha))[::-1]

        #2. calculate lambda = 1/sigma[k]
        lambdak = 1/sigma[k];

        #3. calculate beta = F(lambda, alpha)
        beta = F(alpha,lambdak)

        #4. update alpha = beta - thetaPseudoInv * (theta * beta - b)
        alpha = beta - thetaPseudoInv @ (theta @ beta - b)

        #loop conditions update
        i+=1;
        if(np.linalg.norm(alpha-oldalpha) <= 1e-5):
            break

    #final thresholding step: alpha[i] = 0 if |alpha[i]| <= sigma[k]
    sigma = sorted(np.abs(alpha))[::-1]
    thresh = sigma[k]+1e-8
    for i in range(len(alpha)):
        if abs(alpha[i]) <= thresh:
            alpha[i] = 0

    return alpha
