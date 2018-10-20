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

/*
Kernel implementing part of the Euclidean norm of a vector.
Computes the square of each element and blockwise parallel reduction sum.
The sum of block i is left in v[i * BLOCK_SIZE].
*/
__global__ void normKernel(double *v, int len){

    __shared__ double SMEM[BLOCK_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = threadIdx.x;

    if(tid < len)
        SMEM[idx] = v[tid] * v[tid];
    else
        SMEM[idx] = 0;
    __syncthreads();

    int step = blockDim.x / 2;
    while(step > 0){
        if(idx < step)
            SMEM[idx] = SMEM[idx] + SMEM[idx+step];
        step /= 2;
        __syncthreads();
    }
    if(idx == 0)
        v[tid] = SMEM[idx];
}

/*
Function calculating the Euclidean norm of a vector
the vector v is destroyed during computation
*/
double vectorNorm(double *v, int len){

    int blocks = ceil(len*1.0/BLOCK_SIZE);

    normKernel<<<blocks,BLOCK_SIZE>>>(v,len);
    CHECK(cudaDeviceSynchronize());

    double blockres,res = 0;
    for(int i=0; i<blocks; i++){
        CHECK(cudaMemcpy(&blockres, v+i*BLOCK_SIZE, sizeof(double), cudaMemcpyDeviceToHost));
        res += blockres;
    }
    return sqrt(res);
}

/*
Function calculating MSE: sum((s - D * alphalimaps)^2)/n
*/
double MSE(double *s, double *D, double *alpha, int n, int m){

    int blocks = ceil(n*1.0/BLOCK_SIZE);
    dim3 dimGrid(blocks,1,1);
    dim3 dimBlock(BLOCK_SIZE,1,1);
    double *limapsS,partialMSE;

    CHECK(cudaMalloc(&limapsS, n*sizeof(double)));

    //Initialize cublas
    double cualpha=1,cubeta=0;
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    //limapsS = D * alpha
    CHECK_CUBLAS(cublasDgemv(cublasHandle, CUBLAS_OP_N, n, m, &cualpha, D, n, alpha, 1, &cubeta, limapsS, 1));

    //limapsS = s - limapsS
    vectorSum<<<dimGrid,dimBlock>>>(1, s, -1, limapsS, limapsS, n);
    CHECK(cudaDeviceSynchronize());

    normKernel<<<dimGrid,dimBlock>>>(limapsS, n);
    CHECK(cudaDeviceSynchronize());

    double MSE = 0;
    for(int i=0; i<blocks; i++){
        CHECK(cudaMemcpy(&partialMSE, limapsS+i*BLOCK_SIZE, sizeof(double), cudaMemcpyDeviceToHost));
        MSE += partialMSE;
    }
    MSE /= n;

    CHECK(cudaFree(limapsS));

    return MSE;
}

#endif
