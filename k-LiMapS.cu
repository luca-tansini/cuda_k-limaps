#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cublas_v2.h"
#include "common.h"

#define BLOCK_SIZE 256

//Compare function required by qsort, descending order
//gives -1 if elem1 > elem2, 1 if elem2 > elem1, 0 if equal
int comp(const void *elem1, const void *elem2) {
    float f = *((float*)elem1);
    float s = *((float*)elem2);
    if (f > s) return -1;
    if (f < s) return 1;
    return 0;
}

//Function implementing the F(lambda) shrinkage: b = F(lambda,a)
__global__ void fShrinkage(float lambda, float *a, float *b, int len){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < len){
        b[tid] = a[tid] * (1 - powf(M_E, -1*fabsf(a[tid])));
    }
}

//Function implementing: res = alpha * a + beta * b
__global__ void vectorSum(float alpha, float *a, float beta, float *b, float *res, int len){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < len)
        res[tid] = alpha * a[tid] + beta * b[tid];
}

//Function implementing the final thresholding step
__global__ void thresholding(float *v, int len, float threshold){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < len)
        if(abs(v[tid]) < threshold)
            v[tid] = 0;
}

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
    float cuAlpha = 1, cuBeta = 0;
    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, m, n, &cuAlpha, d_thetaPseudoInv, m, d_b, 1, &cuBeta, d_alpha, 1));

    //algorithm internal loop
    float sigma[m];
    float *d_beta;
    CHECK(cudaMalloc(&d_beta, m*sizeof(float)));
    int i = 0;
    dim3 dimBlock(BLOCK_SIZE,1,1);
    dim3 dimGridM(ceil(m/BLOCK_SIZE),1,1);
    dim3 dimGridN(ceil(n/BLOCK_SIZE),1,1);

    while(i < maxIter){

        //1a. retrieve alpha into sigma
        CHECK_CUBLAS(cublasGetVector(m, sizeof(float), d_alpha, 1, sigma, 1));
        //1b. sort sigma in descending order
        qsort(sigma, m, sizeof(float), comp);

        //2. calculate lambda = 1/sigma[k]
        float lambda = 1/sigma[k];

        //3. calculate beta = F(lambda, alpha)
        fShrinkage<<<dimGridM,dimBlock>>>(lambda, d_alpha, d_beta, m);
        CHECK(cudaDeviceSynchronize());

        //4. update alpha = beta - thetaPseudoInv * (theta * beta - b)
        //using aplha for intermediate results (alpha has size m and m >> n)

        //alpha = theta * beta (€ R^n)
        CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, n, m, &cuAlpha, d_theta, n, d_beta, 1, &cuBeta, d_alpha, 1));

        //alpha = alpha - b (€ R^n)
        vectorSum<<<dimGridN,dimBlock>>>(1, d_alpha, -1, d_b, d_alpha, n);
        CHECK(cudaDeviceSynchronize());

        //alpha = thetaPseudoInv * alpha (€ R^m)
        CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, m, n, &cuAlpha, d_thetaPseudoInv, m, d_alpha, 1, &cuBeta, d_alpha, 1));

        //alpha = beta - alpha (€ R^m)
        vectorSum<<<dimGridM,dimBlock>>>(1, d_beta, -1, d_alpha, d_alpha, m);
        CHECK(cudaDeviceSynchronize());

        //loop conditions update
        i++;
    }

    //final thresholding step: alpha[i] = 0 if |alpha[i]| <= sigma[k]
    thresholding<<<dimGridM,dimBlock>>>(d_alpha, m, sigma[k]);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(alpha, d_alpha, m*sizeof(float), cudaMemcpyHostToDevice));

}
