#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "vectorUtility.cu"

#ifndef _COMMON_H
    #include "common.h"
#endif

#define BLOCK_SIZE 256

//Compare function required by qsort, descending order
//gives -1 if elem1 > elem2, 1 if elem2 > elem1, 0 if equal
int comp(const void *elem1, const void *elem2) {
    double f = *((double*)elem1);
    double s = *((double*)elem2);
    if (f > s) return -1;
    if (f < s) return 1;
    return 0;
}

//Kernel implementing the F(lambda) shrinkage: b = F(lambda,a)
__global__ void fShrinkage(double lambda, double *a, double *b, int len){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < len){
        b[tid] = a[tid] * (1 - powf(M_E, -lambda*fabsf(a[tid])));
    }
}

//Kernel implementing the final thresholding step
__global__ void thresholding(double *v, int len, double threshold){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < len)
        if(fabsf(v[tid]) < threshold)
            v[tid] = 0;
}

/*
Function implementing the k-LiMapS algorithm.
Parameters description:
    k: sparsity level
    D: the dictionary € R^(n*m)
    DINV: the pseudo inverse of the dictionary (€ R^(m*n))
    s: the signal € R^n
    alpha: the output vector € R^m
    maxIter: a max iteration limit for the internal loop
All matrices are required to be in column-major format for compatibility with the CUBLAS libraries
The result is an aproximate solution for s = D*alpha s.t. lzero-norm(alpha) <= k
*/
void devMemK_LiMapS(int k, double *D, int n, int m, double *DINV, double *s, double *alpha, int maxIter){

    //Create the cublas handle
    cublasHandle_t handle;
	CHECK_CUBLAS(cublasCreate(&handle));

    //calculate initial alpha = D * s
    double cuAlpha = 1, cuBeta = 0;
    CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_N, m, n, &cuAlpha, DINV, m, s, 1, &cuBeta, alpha, 1));

    //algorithm internal loop
    int i = 0;
    int mBlocks = ceil(m*1.0/BLOCK_SIZE);
    double *sigma,*beta,*oldalpha,*tmp;
    dim3 dimBlock(BLOCK_SIZE,1,1);
    dim3 dimGridM(mBlocks,1,1);
    dim3 dimGridN(ceil(n*1.0/BLOCK_SIZE),1,1);

    CHECK(cudaMalloc(&beta, m*sizeof(double)));
    CHECK(cudaMalloc(&oldalpha, mBlocks*BLOCK_SIZE*sizeof(double)));
    CHECK(cudaMemset(oldalpha, 0, mBlocks*BLOCK_SIZE*sizeof(double)));
    CHECK(cudaMalloc(&tmp, m*sizeof(double)));
    CHECK(cudaMallocHost(&sigma, m*sizeof(double)));

    while(i < maxIter){

        //1a. retrieve alpha into sigma
        CHECK_CUBLAS(cublasGetVector(m, sizeof(double), alpha, 1, sigma, 1));
        //1b. sort absolute values of sigma in descending order
        for(int j=0; j<m; j++)
            sigma[j] = fabs(sigma[j]);
        qsort(sigma, m, sizeof(double), comp);

        //2. calculate lambda = 1/sigma[k]
        double lambda = 1/sigma[k];

        //3. calculate beta = F(lambda, alpha)
        fShrinkage<<<dimGridM,dimBlock>>>(lambda, alpha, beta, m);
        CHECK(cudaDeviceSynchronize());

        //4. update alpha = beta - DINV * (D * beta - s)
        //using aplha for intermediate results (alpha has size m and m >> n)

        //save oldalpha
        CHECK(cudaMemcpy(oldalpha, alpha, m*sizeof(double), cudaMemcpyDeviceToDevice));

        //alpha = D * beta (€ R^n)
        CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_N, n, m, &cuAlpha, D, n, beta, 1, &cuBeta, alpha, 1));

        //alpha = alpha - s (€ R^n)
        vectorSum<<<dimGridN,dimBlock>>>(1, alpha, -1, s, alpha, n);
        CHECK(cudaDeviceSynchronize());

        //tmp = DINV * alpha (€ R^m)
        CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_N, m, n, &cuAlpha, DINV, m, alpha, 1, &cuBeta, tmp, 1));

        //alpha = beta - tmp (€ R^m)
        vectorSum<<<dimGridM,dimBlock>>>(1, beta, -1, tmp, alpha, m);
        CHECK(cudaDeviceSynchronize());

        //loop conditions update
        vectorSum<<<dimGridM,dimBlock>>>(1, alpha, -1, oldalpha, oldalpha, m);
        CHECK(cudaDeviceSynchronize());
        double norm = vectorNorm(oldalpha, m);
        if(norm <= 1e-5){
            break;
        }
        i++;
    }

    //final thresholding step: alpha[i] = 0 if |alpha[i]| <= sigma[k]
    thresholding<<<dimGridM,dimBlock>>>(alpha, m, sigma[k]);
    CHECK(cudaDeviceSynchronize());

    //Free Memory
    CHECK(cudaFree(oldalpha));
    CHECK(cudaFree(beta));
    CHECK(cudaFree(tmp));
    CHECK(cudaFreeHost(sigma));
    CHECK_CUBLAS(cublasDestroy(handle));

}
