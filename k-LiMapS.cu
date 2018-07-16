#include <stdio.h>
#include "cublas_v2.h"
#include "../common.h"

/*
Function implementing the k-LiMapS algorithm.
Parameters description:
    k: sparsity level
    theta: the dictionary € R^(n*m)
    thetaPseudoInv: the pseudo inverse of the dictionary (€ R^(m*n))
    b: the signal € R^n
    alpha: the output vector € R^m
    maxIter: a max iteration limit for the internal loop
All matrices are required to be in column-major format for compatibility with the CUBLAS libraries
The result is an aproximate solution for b = theta*alpha s.t. lzero-norm(alpha) <= k
*/
void k_LiMapS(int k, float *theta, int n, int m, float *thetaPseudoInv, float *b, float *alpha, int maxIter){

    //Create the cublas handle
    cublasHandle_t handle;
	CHECK_CUBLAS(cublasCreate(&handle));

    //device memory pointers and allocation
    float *d_theta, *d_thetaPseudoInv, *d_b, *d_alpha;

    CHECK(cudaMalloc(&d_theta, n*m*sizeof(float)));
    CHECK(cudaMalloc(&d_thetaPseudoInv, m*n*sizeof(float)));
    CHECK(cudaMalloc(&d_b, n*sizeof(float)));
    CHECK(cudaMalloc(&d_alpha, m*sizeof(float)));

    //transfer dictionary and pseudoinverse to device memory
    CHECK_CUBLAS(cublasSetMatrix(n, m, sizeof(float), theta, n, d_theta, n));
    CHECK_CUBLAS(cublasSetMatrix(m, n, sizeof(float), thetaPseudoInv, m, d_thetaPseudoInv, m));
    CHECK_CUBLAS(cublasSetVector(n, sizeof(float), b, 1, d_b, 1));
    CHECK(cudaMemset(d_alpha, 0, m*sizeof(float)));

    //calculate initial alpha = thetaPseudoInv * b
    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, m, n, 1, d_thetaPseudoInv, m, d_b, 1, 0, d_alpha, 1));

    //algorithm internal loop
    while(/*conditions*/){

        //1a. retrieve alpha into sigma
        //1b. sort sigma in descending order

        //2. calculate lambda = 1/sigma[k]

        //3. calculate beta = F(lambda, alpha)

        //4. update alpha = beta - thetaPseudoInv * (theta * beta - b)

    }

    //final thresholding step: alpha[i] = 0 if |alpha[i]| <= sigma[k]

}
