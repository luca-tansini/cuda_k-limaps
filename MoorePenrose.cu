#include <float.h>
#include <cusolverDn.h>

/*
Kernel that takes the vector of singular values S and produces the pseudo inverse of the diagonal matrix of S. According to the instances, we assume n < m. Since [n x m] would be the dimension of S diag matrix, and we also have to transpose the diag matrix, its pseudo inverse will be [m x n], with leading dimension m.
The elemnts on the main diagonal are to be inverted only if non-zero. To determine what is zero we use a threshold based on the machine FLT_EPSILON.
*/
__global__ void calculateDiagPseudoInv(float *S, float *SPseudoInv, int n, int m){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < n)
        if(fabsf(S[idx]) > FLT_EPSILON)
            SPseudoInv[idx*m + idx] = 1/S[idx];
        else
            SPseudoInv[idx*m + idx] = S[idx];
}

/*
This function calcuates the Moore-Penrose inverse matrix of the input matrix A (n*m, with m > n), leaving the result in apseudoinv, assumed preallocated.

The pseudoinverse is computed via SVD.
If SVD(A) = U*S*V^T --> A^+ = V * S^+ * U^T, where S^+ is obtained replacing each non-zero element on the diagonal with its reciprocal.

The cuSOLVER libraries used to calculate the SVD need the input matrix to be n x m with n < m, so we need to transpose our matrix.

According to math
being SVD(A) = U * S * V_T,
being SVD(A^T) = U1 * S1 * V_T1,
we have that
U == V_T1^T (at least in absolute values)
S == S1
V_T == U1^T (at least in absolute values)
A == U * S * V_T == V_T1^T * S1 * U1^T == A

So we can compute A^+ as
U1^T * S1^+ * V_T1
*/
void MoorePenroseInverse(float *A, int n, int m, float *APseudoInv){

    cusolverDnHandle_t cusolverHandle;
    cublasHandle_t cublasHandle;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverHandle));
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    //transpose matrix A into APseudoInv
    float alpha=1,beta=0;
    CHECK_CUBLAS(cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &alpha, A, m, &beta, A, n, APseudoInv, n));

    //Get dimension needed for the workspace buffer and allocate it
    int bufferDim;
    float *buffer;
    CHECK_CUSOLVER(cusolverDnSgesvd_bufferSize(cusolverHandle, m, n, &bufferDim));
    CHECK(cudaMalloc(&buffer,bufferDim));

    //Allocate U1,S1,V_T1
    float *U1,*S1,*V_T1;
    CHECK(cudaMalloc(&U1, m*m*sizeof(float)));
    CHECK(cudaMalloc(&S1, n*sizeof(float)));
    CHECK(cudaMalloc(&V_T1, n*n*sizeof(float)));

    //Calculate SVD with cuSOLVER
    int *dev_info, h_dev_info;
    CHECK(cudaMalloc(&dev_info, sizeof(int)));
    CHECK_CUSOLVER(cusolverDnSgesvd(cusolverHandle, 'A', 'A', m, n, APseudoInv, m, S1, U1, m, V_T1, n, buffer, bufferDim, NULL, dev_info));
    CHECK(cudaMemcpy(&h_dev_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost));
    if(h_dev_info != 0)
        printf("Something went wrong (dev_info=%d)\n", h_dev_info);

    //Calculate S1^+
    float *S1PseudoInv;
    CHECK(cudaMalloc(&S1PseudoInv, m*n*sizeof(float)));
    CHECK(cudaMemset(S1PseudoInv, 0, m*n*sizeof(float)));

    dim3 dimBlock(BLOCK_SIZE,1,1);
    dim3 dimGrid(ceil(n*1.0/BLOCK_SIZE),1,1);
    calculateDiagPseudoInv<<<dimGrid,dimBlock>>>(S1, S1PseudoInv, n, m);
    CHECK(cudaDeviceSynchronize());

    //calculate APseudoInv = U1^T * S1^+ * V_T1
    //APseudoInv = U1^T * S1^+
    CHECK_CUBLAS(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, m, &alpha, U1, m, S1PseudoInv, m, &beta, APseudoInv, m));
    //APseudoInv *= V_T1
    CHECK_CUBLAS(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, &alpha, APseudoInv, m, V_T1, n, &beta, APseudoInv, m));

    //Free memory
    CHECK(cudaFree(buffer));
    CHECK(cudaFree(U1));
    CHECK(cudaFree(S1));
    CHECK(cudaFree(S1PseudoInv));
    CHECK(cudaFree(V_T1));

}
