#include <float.h>
#include <cusolverDn.h>
#include "singular_value_decomposition.h"

#ifndef _COMMON_H
    #include "common.h"
#endif

#ifndef _MATRIX_PRINT_H
    #include "matrixPrint.h"
#endif

#define BLOCK_SIZE 256

/*
Kernel that takes the vector of singular values S (of length m) and produces the pseudo inverse of the diagonal matrix of S. Since [n x m] would be the dimension of S diag matrix, and we also have to transpose the diag matrix, its pseudo inverse will be [m x n], with leading dimension m.
The elements on the main diagonal are to be inverted only if non-zero. To determine what is zero we use a threshold based on the machine DBL_EPSILON.
*/
__global__ void calculateDiagPinv(double *S, double *Spinv, int n, int m){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < m && idx < n){
        if(fabsf(S[idx]) > DBL_EPSILON)
            Spinv[idx*m + idx] = 1/S[idx];
        else
            Spinv[idx*m + idx] = 0;
    }
}

/*
This version uses cusolverDnSgesvdj which uses the Jacobi method for SVD decomposition.
There is no constraint on n and m.
cusolverDnSgesvdj returns V instead of VH
*/
void JacobiMoorePenroseInverse(double *A, int n, int m, double *Apinv){

    cusolverDnHandle_t cusolverHandle;
    cublasHandle_t cublasHandle;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverHandle));
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    //Allocate U,S,V
    double *U,*S,*V;
    CHECK(cudaMalloc(&U, n*n*sizeof(double)));
    CHECK(cudaMalloc(&S, m*sizeof(double)));
    CHECK(cudaMalloc(&V, m*m*sizeof(double)));

    //Calculate SVD with cuSOLVER
    double *Acopy; //we use a copy of A because gesvdj destroys input matrix
    CHECK(cudaMalloc(&Acopy, n*m*sizeof(double)));
    CHECK(cudaMemcpy(Acopy, A, n*m*sizeof(double), cudaMemcpyDeviceToDevice));

    //Set up cusolverDnDgesvdj parameters
    int bufferDim;
    double *buffer;
    gesvdjInfo_t gesvdj_params = NULL; //deafult params
    cusolverDnCreateGesvdjInfo(&gesvdj_params);
    CHECK_CUSOLVER(cusolverDnDgesvdj_bufferSize(cusolverHandle, CUSOLVER_EIG_MODE_VECTOR, 0, n, m, Acopy, n, S, U, n, V, m, &bufferDim, gesvdj_params));
    CHECK(cudaMalloc(&buffer,bufferDim*sizeof(double)));

    //Call cusolverDnDgesvdj
    int *dev_info, h_dev_info;
    CHECK(cudaMalloc(&dev_info, sizeof(int)));
    CHECK_CUSOLVER(cusolverDnDgesvdj(cusolverHandle, CUSOLVER_EIG_MODE_VECTOR, 0, n, m, Acopy, n, S, U, n, V, m, buffer, bufferDim, dev_info, gesvdj_params));
    CHECK(cudaMemcpy(&h_dev_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost));
    if(h_dev_info != 0)
        printf("Something went wrong (dev_info=%d)\n", h_dev_info);

    //Calculate S^+
    double *Spinv;
    CHECK(cudaMalloc(&Spinv, m*n*sizeof(double)));
    CHECK(cudaMemset(Spinv, 0, m*n*sizeof(double)));

    dim3 dimBlock(BLOCK_SIZE,1,1);
    dim3 dimGrid(ceil(m*1.0/BLOCK_SIZE),1,1);
    calculateDiagPinv<<<dimGrid,dimBlock>>>(S, Spinv, n, m);
    CHECK(cudaDeviceSynchronize());

    //calculate Apinv = V * S^+ * U^T
    double alpha=1,beta=0,*tmp;
    CHECK(cudaMalloc(&tmp, m*n*sizeof(double)));

    //tmp = V * S^+
    CHECK_CUBLAS(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, m, &alpha, V, m, Spinv, m, &beta, tmp, m));

    //Apinv = tmp * U^T
    CHECK_CUBLAS(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, n, &alpha, tmp, m, U, n, &beta, Apinv, m));

    //Free memory
    CHECK(cudaFree(buffer));
    CHECK(cudaFree(U));
    CHECK(cudaFree(S));
    CHECK(cudaFree(Spinv));
    CHECK(cudaFree(V));
    CHECK(cudaFree(tmp));
    CHECK_CUBLAS(cublasDestroy(cublasHandle));
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverHandle));

}

/*
This function calcuates the Moore-Penrose inverse matrix of the input matrix A (n*m, with n > m), leaving the result in Apinv, assumed preallocated.

The pseudoinverse is computed via SVD.
If SVD(A) = U*S*V^T --> A^+ = V * S^+ * U^T, where S^+ is obtained replacing each non-zero element on the diagonal with its reciprocal and transposing.

The cuSOLVER libraries used to calculate the SVD need the input matrix to be n x m with n >= m.
*/
void MoorePenroseInverse(double *A, int n, int m, double *Apinv){

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
    double *buffer;
    CHECK_CUSOLVER(cusolverDnDgesvd_bufferSize(cusolverHandle, n, m, &bufferDim));
    CHECK(cudaMalloc(&buffer,bufferDim*sizeof(double)));

    //Allocate U,S,VT
    double *U,*S,*VT;
    CHECK(cudaMalloc(&U, n*n*sizeof(double)));
    CHECK(cudaMalloc(&S, m*sizeof(double)));
    CHECK(cudaMalloc(&VT, m*m*sizeof(double)));

    //Calculate SVD with cuSOLVER
    double *Acopy; //we use a copy of A because apparently gesvd destroys input matrix
    CHECK(cudaMalloc(&Acopy, n*m*sizeof(double)));
    CHECK(cudaMemcpy(Acopy, A, n*m*sizeof(double), cudaMemcpyDeviceToDevice));

    int *dev_info, h_dev_info;
    CHECK(cudaMalloc(&dev_info, sizeof(int)));
    CHECK_CUSOLVER(cusolverDnDgesvd(cusolverHandle, 'A', 'A', n, m, Acopy, n, S, U, n, VT, m, buffer, bufferDim, NULL, dev_info));
    CHECK(cudaMemcpy(&h_dev_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost));
    if(h_dev_info != 0)
        printf("Something went wrong (dev_info=%d)\n", h_dev_info);

    //Calculate S^+
    double *Spinv;
    CHECK(cudaMalloc(&Spinv, m*n*sizeof(double)));
    CHECK(cudaMemset(Spinv, 0, m*n*sizeof(double)));

    dim3 dimBlock(BLOCK_SIZE,1,1);
    dim3 dimGrid(ceil(m*1.0/BLOCK_SIZE),1,1);
    calculateDiagPinv<<<dimGrid,dimBlock>>>(S, Spinv, n, m);
    CHECK(cudaDeviceSynchronize());

    //calculate Apinv = VT^T * S^+ * U^T
    double alpha=1,beta=0,*tmp;
    CHECK(cudaMalloc(&tmp, m*n*sizeof(double)));

    //tmp = VT^T * S^+
    CHECK_CUBLAS(cublasDgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, m, &alpha, VT, m, Spinv, m, &beta, tmp, m));

    //Apinv = tmp * U^T
    CHECK_CUBLAS(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, n, &alpha, tmp, m, U, n, &beta, Apinv, m));

    //Free memory
    CHECK(cudaFree(buffer));
    CHECK(cudaFree(U));
    CHECK(cudaFree(S));
    CHECK(cudaFree(Spinv));
    CHECK(cudaFree(VT));
    CHECK(cudaFree(tmp));
    CHECK_CUBLAS(cublasDestroy(cublasHandle));
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverHandle));

}

void TransposedMoorePenroseInverse(double *A, int n, int m, double *Apinv){

    double *AT,*ApinvT;
    CHECK(cudaMalloc(&AT, m*n*sizeof(double)));
    CHECK(cudaMalloc(&ApinvT, n*m*sizeof(double)));

    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    double alpha = 1, beta = 0;

    //Transpose A
    CHECK_CUBLAS(cublasDgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, A, n, &beta, A, n, AT, m));

    //Call MoorePenroseInverse
    MoorePenroseInverse(AT, m, n, ApinvT);

    //Transpose ApinvT
    CHECK_CUBLAS(cublasDgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, ApinvT, n, &beta, ApinvT, n, Apinv, m));

    CHECK(cudaFree(AT));
    CHECK(cudaFree(ApinvT));
    CHECK_CUBLAS(cublasDestroy(cublasHandle));

}

//The performed check is A * Apinv * A =?= A
int CheckPseudoinverse(double *A, int n, int m, double *Apinv){

    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    //tmp = A * Apinv
    double *tmp,alpha=1,beta=0;
    CHECK(cudaMalloc(&tmp, n*n*sizeof(double)));

    CHECK_CUBLAS(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, m, &alpha, A, n, Apinv, m, &beta, tmp, n));

    //tmp2 = tmp * A
    double *tmp2;
    CHECK(cudaMalloc(&tmp2, n*m*sizeof(double)));

    CHECK_CUBLAS(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, n, &alpha, tmp, n, A, n, &beta, tmp2, n));

    //tmp2 =?= A
    double *h_tmp2,*h_A;
    CHECK(cudaMallocHost(&h_tmp2, n*m*sizeof(double)));
    CHECK(cudaMemcpy(h_tmp2, tmp2, n*m*sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMallocHost(&h_A, n*m*sizeof(double)));
    CHECK(cudaMemcpy(h_A, A, n*m*sizeof(double), cudaMemcpyDeviceToHost));

    int i;
    for(i=0; i<n*m; i++)
        if(fabs(h_A[i] - h_tmp2[i]) > 1e-5){
            printf("at index %d diff is: %f\n",i, h_A[i] - h_tmp2[i]);
            break;
        }

    int ret=1;
    if(i < n*m)
        ret = 0;

    CHECK(cudaFree(tmp));
    CHECK(cudaFree(tmp2));
    CHECK(cudaFreeHost(h_tmp2));
    CHECK(cudaFreeHost(h_A));
    CHECK_CUBLAS(cublasDestroy(cublasHandle));

    return ret;
}

//Function that computes MoorePenrose pseudoinverse using host libraries.
//The library assumes the input matrix are RowMajor n x m with n >= m
//We use our ColMajor matrices as input which conveniently when read RowMajor with n and m switched are exactly their own transposed, with m >= n.
//This is possible because the pseudoinverse of the transposed is the transposed of the pseudoinverse.
//The n and m switch is performed inside the procedure.
void HostMoorePenroseInverse(double *d_A, int n, int m, double *d_Apinv){

    int nrows = m;
    int ncols = n;
    double *A,*Apinv,*U,*VT,*S,*dummy_array;

    CHECK(cudaMallocHost(&A, n*m*sizeof(double)));
    CHECK(cudaMallocHost(&Apinv, m*n*sizeof(double)));
    CHECK(cudaMemcpy(A, d_A, n*m*sizeof(double), cudaMemcpyDeviceToHost));

    dummy_array = (double*) malloc(ncols * sizeof(double));
    if(dummy_array == NULL){ printf(" No memory available\n"); exit(0);}

    U = (double *) malloc(nrows * ncols * sizeof(double));
    if(U == NULL){ printf(" No memory available\n"); exit(0);}

    S = (double *) malloc(ncols * sizeof(double));
    if(S == NULL){ printf(" No memory available\n"); exit(0);}

    VT = (double *) malloc(ncols * ncols * sizeof(double));
    if(VT == NULL){ printf(" No memory available\n"); exit(0);}

    int err = Singular_Value_Decomposition(A, nrows, ncols, U, S, VT, dummy_array);

    if(err < 0)
        printf(" Failed to converge\n");

    Singular_Value_Decomposition_Inverse(U, S, VT, 0, nrows, ncols, Apinv);

    CHECK(cudaMemcpy(d_Apinv, Apinv, m*n*sizeof(double), cudaMemcpyHostToDevice));

    free(dummy_array);
    CHECK(cudaFreeHost(A));
    CHECK(cudaFreeHost(Apinv));

}
