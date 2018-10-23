#include "common.h"
#include <curand.h>
#include <curand_kernel.h>
#define BLOCK_SIZE 256

#ifndef _VECTOR_UTIL_CU_
#define _VECTOR_UTIL_CU_

/*
Kernel che implementa la somma di due vettori, opportunamente scalati: res = alpha * a + beta * b
*/
__global__ void vectorSum(double alpha, double *a, double beta, double *b, double *res, int len){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < len)
        res[tid] = alpha * a[tid] + beta * b[tid];
}

/*
Kernel che implementa parte del calcolo della norma di un vettore, nello specifico eleva al quadrato tutti gli elementi e fa una parallel reduction sum.
La somma di ogni blocco i viene messa in v[i * BLOCK_SIZE].
IMPORTANTE: questo kernel altera il contenuto del vettore in input!
*/
__global__ void normKernel(double *v, int len){

    __shared__ double SMEM[BLOCK_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = threadIdx.x;

    //Carica il quadrato in shared memory
    if(tid < len)
        SMEM[idx] = v[tid] * v[tid];
    else
        SMEM[idx] = 0;
    __syncthreads();

    //Parallel reduction
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
Funzione per il calcolo della norma euclidea di un vettore.
Il vettore v viene distrutto durante la computazione!
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
Funzione per il calcolo del MSE: sum((s - D * alphalimaps)^2)/n
*/
double MSE(double *s, double *D, double *alpha, int n, int m){

    int blocks = ceil(n*1.0/BLOCK_SIZE);
    dim3 dimGrid(blocks,1,1);
    dim3 dimBlock(BLOCK_SIZE,1,1);
    double *limapsS,partialMSE;

    CHECK(cudaMalloc(&limapsS, n*sizeof(double)));

    //Inizializza cublas
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

/*
Kernel che utilizza le librerie cuRAND per riempire un vettore di valori estratti da una distribuzione gaussiana.
*/
__global__ void normfill(double *v, int len, curandState *states, int seed){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < len){
        curand_init(tid*seed+seed, 0, 0, &states[tid]);
        v[tid] = curand_normal_double(&states[tid]);
    }

}

/*
Kernel che effettua la divisione di ogni elemento del vettore v per x.
IMPORTANTE: eventuali controlli su x != 0 vanno effettuati prima della chiamata!
*/
__global__ void divide(double *v, double x, int len){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < len)
        v[tid] /= x;

}

#endif
