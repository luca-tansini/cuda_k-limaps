#include <float.h>
#include <cusolverDn.h>

#ifndef _COMMON_H
    #include "common.h"
#endif

#ifndef _MATRIX_PRINT_H
    #include "matrixPrint.h"
#endif

#define BLOCK_SIZE 256

/*
Kernel that takes the vector of singular values S (of length m) and produces the pseudo inverse of the diagonal matrix of S. According to the instances, we assume n >= m. Since [n x m] would be the dimension of S diag matrix, and we also have to transpose the diag matrix, its pseudo inverse will be [m x n], with leading dimension m.
The elements on the main diagonal are to be inverted only if non-zero. To determine what is zero we use a threshold based on the machine FLT_EPSILON.
*/
__global__ void calculateDiagPseudoInv(float *S, float *SPseudoInv, int n, int m){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < m){
        if(fabsf(S[idx]) > FLT_EPSILON)
            SPseudoInv[idx*m + idx] = 1/S[idx];
        else
            SPseudoInv[idx*m + idx] = S[idx];
    }
}

/*
This function calcuates the Moore-Penrose inverse matrix of the input matrix A (n*m, with n > m), leaving the result in APseudoInv, assumed preallocated.

The pseudoinverse is computed via SVD.
If SVD(A) = U*S*V^T --> A^+ = V * S^+ * U^T, where S^+ is obtained replacing each non-zero element on the diagonal with its reciprocal and transposing.

The cuSOLVER libraries used to calculate the SVD need the input matrix to be n x m with n >= m.
*/
void MoorePenroseInverse(float *A, int n, int m, float *APseudoInv){

    if(n < m){
        printf("error: n must be >= m! (you can transpose the input matrix and then transpose the result to work with matrices that have less rows than columns)\n");
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
    float *U,*S,*V_T;
    CHECK(cudaMalloc(&U, n*n*sizeof(float)));
    CHECK(cudaMalloc(&S, m*sizeof(float)));
    CHECK(cudaMalloc(&V_T, m*m*sizeof(float)));

    //Calculate SVD with cuSOLVER

    float *Acopy; //we use a copy of A because apparently gesvd destroys input matrix
    CHECK(cudaMalloc(&Acopy, n*m*sizeof(float)));
    CHECK(cudaMemcpy(Acopy, A, n*m*sizeof(float), cudaMemcpyDeviceToDevice));

    int *dev_info, h_dev_info;
    CHECK(cudaMalloc(&dev_info, sizeof(int)));
    CHECK_CUSOLVER(cusolverDnSgesvd(cusolverHandle, 'A', 'A', n, m, Acopy, n, S, U, n, V_T, m, buffer, bufferDim, NULL, dev_info));
    CHECK(cudaMemcpy(&h_dev_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost));
    if(h_dev_info != 0)
        printf("Something went wrong (dev_info=%d)\n", h_dev_info);

    //Calculate S^+
    float *SPseudoInv;
    CHECK(cudaMalloc(&SPseudoInv, m*n*sizeof(float)));
    CHECK(cudaMemset(SPseudoInv, 0, m*n*sizeof(float)));

    dim3 dimBlock(BLOCK_SIZE,1,1);
    dim3 dimGrid(ceil(m*1.0/BLOCK_SIZE),1,1);
    calculateDiagPseudoInv<<<dimGrid,dimBlock>>>(S, SPseudoInv, n, m);
    CHECK(cudaDeviceSynchronize());

    //calculate APseudoInv = V_T^T * S^+ * U^T
    //APseudoInv = V_T^T * S^+
    float alpha=1,beta=0;
    CHECK_CUBLAS(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, m, &alpha, V_T, m, SPseudoInv, m, &beta, APseudoInv, m));
    //APseudoInv *= U^T
    CHECK_CUBLAS(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, n, &alpha, APseudoInv, m, U, n, &beta, APseudoInv, m));

    //Free memory
    CHECK(cudaFree(buffer));
    CHECK(cudaFree(U));
    CHECK(cudaFree(S));
    CHECK(cudaFree(SPseudoInv));
    CHECK(cudaFree(V_T));
    CHECK_CUBLAS(cublasDestroy(cublasHandle));
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverHandle));

}

void TransposedMoorePenroseInverse(float *A, int n, int m, float *APseudoInv){

    float *AT,*APseudoInvT;
    CHECK(cudaMalloc(&AT, m*n*sizeof(float)));
    CHECK(cudaMalloc(&APseudoInvT, n*m*sizeof(float)));

    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    float alpha = 1, beta = 0;

    //Transpose A
    CHECK_CUBLAS(cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, A, n, &beta, A, n, AT, m));

    //Call MoorePenroseInverse
    MoorePenroseInverse(AT, m, n, APseudoInvT);

    //Transpose APseudoInvT
    CHECK_CUBLAS(cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, APseudoInvT, n, &beta, APseudoInvT, n, APseudoInv, m));

    CHECK(cudaFree(AT));
    CHECK(cudaFree(APseudoInvT));
    CHECK_CUBLAS(cublasDestroy(cublasHandle));

}

int CheckPseudoInverse(float *A, int n, int m, float *Apinv){

    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    float *id,*d_id,alpha=1,beta=0;
    CHECK(cudaMalloc(&d_id, m*m*sizeof(float)));

    CHECK_CUBLAS(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, m, n, &alpha, Apinv, m, A, n, &beta, d_id, m));

    CHECK(cudaMallocHost(&id, m*m*sizeof(float)));
    CHECK(cudaMemcpy(id, d_id, m*m*sizeof(float), cudaMemcpyDeviceToHost));

    int i;
    for(i=0;i<m;i++)
        for(int j=0; j<m; j++){
            if(i == j && abs(1-id[j*m+i]) > 1e-4) break;
            if(i != j && abs(0-id[j*m+i]) > 1e-4) break;
        }

    int ret;
    if(i < m){
        printf("Apinv * A:\n");
        printColumnMajorMatrix(id, m, m);
        ret = 0;
    }
    else{
        ret = 1;
    }

    CHECK(cudaFree(d_id));
    CHECK(cudaFreeHost(id));
    CHECK_CUBLAS(cublasDestroy(cublasHandle));
    return ret;
}
