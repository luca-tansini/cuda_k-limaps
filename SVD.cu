#include <float.h>
#include <cusolverDn.h>
#include "common.h"
#include "matrixPrint.h"

#define BLOCK_SIZE 256

__global__ void calculateDiagS(float *s, float *S, int n, int m){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < m)
        S[idx*n + idx] = s[idx];
}

void SVDRebuild(float *A, int n, int m, float *A2){

    if(n < m){
        printf("error: n must be >= m!\n");
        return;
    }

    cusolverDnHandle_t cusolverHandle;
    cublasHandle_t cublasHandle;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverHandle));
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    //Get dimension needed for the workspace buffer and allocate it
    int bufferDim;
    float *buffer;
    CHECK_CUSOLVER(cusolverDnSgesvd_bufferSize(cusolverHandle, n, m, &bufferDim));
    CHECK(cudaMalloc(&buffer,bufferDim*sizeof(float)));

    //Allocate U,S,V_T
    float *U,*s,*S,*V_T;
    CHECK(cudaMalloc(&U, n*n*sizeof(float)));
    CHECK(cudaMalloc(&s, m*sizeof(float)));
    CHECK(cudaMalloc(&S, n*m*sizeof(float)));
    CHECK(cudaMalloc(&V_T, m*m*sizeof(float)));

    //Calculate SVD with cuSOLVER
    int *dev_info, h_dev_info;
    CHECK(cudaMalloc(&dev_info, sizeof(int)));
    CHECK_CUSOLVER(cusolverDnSgesvd(cusolverHandle, 'A', 'A', n, m, A, n, s, U, n, V_T, m, buffer, bufferDim, NULL, dev_info));
    CHECK(cudaMemcpy(&h_dev_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost));
    if(h_dev_info != 0)
        printf("Something went wrong (dev_info=%d)\n", h_dev_info);

    //Calculate S
    CHECK(cudaMemset(S, 0, n*m*sizeof(float)));
    dim3 dimBlock(BLOCK_SIZE,1,1);
    dim3 dimGrid(ceil(m*1.0/BLOCK_SIZE),1,1);
    calculateDiagS<<<dimGrid,dimBlock>>>(s, S, n, m);
    CHECK(cudaDeviceSynchronize());

    //calculate A2 = U * S * V_T
    float alpha, beta;
    alpha = 1;
    beta = 0;

    //A2 = U * S
    CHECK_CUBLAS(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, n, &alpha, U, n, S, n, &beta, A2, n));

    //A2 *= V_T
    CHECK_CUBLAS(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, m, &alpha, A2, n, V_T, m, &beta, A2, n));

    //Free memory
    CHECK(cudaFree(buffer));
    CHECK(cudaFree(U));
    CHECK(cudaFree(s));
    CHECK(cudaFree(S));
    CHECK(cudaFree(V_T));

}
