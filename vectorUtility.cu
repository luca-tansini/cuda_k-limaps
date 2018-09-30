#include "common.h"
#define BLOCK_SIZE 256

#ifndef _VECTOR_UTIL_CU_
#define _VECTOR_UTIL_CU_



//Kernel implementing: res = alpha * a + beta * b
__global__ void vectorSum(double alpha, double *a, double beta, double *b, double *res, int len){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < len)
        res[tid] = alpha * a[tid] + beta * b[tid];
}

//Kernel implementing the 2-norm of a vector (the vector is destroyed after computation, with v[i] being the partial sum of block i)
__global__ void vector2norm(double *v){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    v[tid] *= v[tid];

    int step = blockDim.x / 2;
    int idx = threadIdx.x;
    double *p = v + blockDim.x * blockIdx.x;
    while(step > 0){
        if(idx < step)
            p[idx] = p[idx] + p[idx+step];
        step /= 2;
        __syncthreads();
    }
    if(idx == 0)
        v[blockIdx.x] = p[idx];
}

/*
Function calculating MSE: sum((s - D * alphalimaps)^2)/n
*/
double MSE(double *s, double *D, double *alpha, int n, int m){

    int blocks = ceil(n*1.0/BLOCK_SIZE);
    dim3 dimGrid(blocks,1,1);
    dim3 dimBlock(BLOCK_SIZE,1,1);
    double *limapsS,*partialMSEBlocks;

    CHECK(cudaMalloc(&limapsS, blocks*BLOCK_SIZE*sizeof(double)));
    CHECK(cudaMemset(limapsS, 0, blocks*BLOCK_SIZE*sizeof(double)));
    CHECK(cudaMallocHost(&partialMSEBlocks, blocks*sizeof(double)));

    //Initialize cublas
    double cualpha=1,cubeta=0;
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    //limapsS = D * alpha
    CHECK_CUBLAS(cublasDgemv(cublasHandle, CUBLAS_OP_N, n, m, &cualpha, D, n, alpha, 1, &cubeta, limapsS, 1));

    //limapsS = s - limapsS
    vectorSum<<<dimGrid,dimBlock>>>(1, s, -1, limapsS, limapsS, n);
    CHECK(cudaDeviceSynchronize());

    vector2norm<<<dimGrid,dimBlock>>>(limapsS);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(partialMSEBlocks, limapsS, blocks * sizeof(double), cudaMemcpyDeviceToHost));
    double MSE = 0;
    for(int j=0; j<blocks; j++)
        MSE += partialMSEBlocks[j];
    MSE /= n;

    CHECK(cudaFree(limapsS));
    CHECK(cudaFreeHost(partialMSEBlocks));

    return MSE;
}

#endif
