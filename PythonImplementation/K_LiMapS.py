import numpy as np

#Calculates MoorePenrose pseudo-inverse of matrix A via SVD
#Being SVD(A) = U * S * V^T, A^+ = V * S^+ * U^T
def MoorePenrosePseudoInverse(A,n,m):

    #Calculate SVD(A)
    U, s, V_T = np.linalg.svd(A, full_matrices=True)

    #Calculate S^+
    S = [ 1/x if abs(x) > np.finfo(float).eps else 0 for x in s]
    SPseudoInv = np.zeros((n,m))
    SPseudoInv[:min(n,m), :min(n,m)] = np.diag(S)
    SPseudoInv = np.matrix(SPseudoInv).transpose()

    return V_T.transpose() * SPseudoInv * U.transpose()


#Function implementing k-LiMapS algorithm
def k_LiMapS(k, theta, thetaPseudoInv, b, maxIter):

    #calculate initial alpha = thetaPseudoInv * b
    alpha = thetaPseudoInv * b

    #algorithm internal loop
    i = 0;
    while(i < maxIter):

        #1b. sort sigma in descending order
        oldalpha = alpha
        alpha = alpha.transpose().tolist()[0]
        sigma = sorted(np.abs(alpha))[::-1]

        #2. calculate lambda = 1/sigma[k]
        lambdaK = 1/sigma[k];

        #3. calculate beta = F(lambda, alpha)
        beta = [x * (1-np.exp(-lambdaK*abs(x))) for x in alpha]
        beta = np.matrix(beta).transpose()

        #4. update alpha = beta - thetaPseudoInv * (theta * beta - b)
        alpha = beta - thetaPseudoInv * (theta * beta - b)

        #loop conditions update
        i+=1;
        if(np.linalg.norm(alpha-oldalpha) < 1e-6):
            #print(i)
            break

    #final thresholding step: alpha[i] = 0 if |alpha[i]| <= sigma[k]
    for i in range(len(alpha)):
        if abs(alpha[i]) <= sigma[k]:
            alpha[i] = 0

    return alpha
