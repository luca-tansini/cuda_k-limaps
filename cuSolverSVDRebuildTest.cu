#include <cusolverDn.h>
#include "common.h"
#include "matrixPrint.h"

#define BLOCK_SIZE 256

/*
Test che cerca di ricostruire la matrice decomposta da cusolverDnSgesvd.
Funziona sia con dimensioni grandi che "strane".
*/

__global__ void buildDiagMatrix(double *s, double *S, int n, int m){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < m)
        S[idx*n + idx] = s[idx];
}

void SVDRebuild(double *A, int n, int m, double *A2){

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
    double *U,*s,*S,*VT;
    CHECK(cudaMalloc(&U, n*n*sizeof(double)));
    CHECK(cudaMalloc(&s, m*sizeof(double)));
    CHECK(cudaMalloc(&S, n*m*sizeof(double)));
    CHECK(cudaMalloc(&VT, m*m*sizeof(double)));

    //Calculate SVD with cuSOLVER
    int *dev_info, h_dev_info;
    CHECK(cudaMalloc(&dev_info, sizeof(int)));
    CHECK_CUSOLVER(cusolverDnDgesvd(cusolverHandle, 'A', 'A', n, m, A, n, s, U, n, VT, m, buffer, bufferDim, NULL, dev_info));
    CHECK(cudaMemcpy(&h_dev_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost));
    if(h_dev_info != 0)
        printf("Something went wrong (dev_info=%d)\n", h_dev_info);

    //build S diagonal matrix
    CHECK(cudaMemset(S, 0, n*m*sizeof(double)));
    dim3 dimBlock(BLOCK_SIZE,1,1);
    dim3 dimGrid(ceil(m*1.0/BLOCK_SIZE),1,1);
    buildDiagMatrix<<<dimGrid,dimBlock>>>(s, S, n, m);
    CHECK(cudaDeviceSynchronize());

    //calculate A2 = U * S * VT
    double alpha, beta, *tmp;
    alpha = 1;
    beta = 0;
    CHECK(cudaMalloc(&tmp, n*m*sizeof(double)));

    //tmp = U * S
    CHECK_CUBLAS(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, n, &alpha, U, n, S, n, &beta, tmp, n));

    //A2 = tmp * V_T
    CHECK_CUBLAS(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, m, &alpha, tmp, n, VT, m, &beta, A2, n));

    //Free memory
    CHECK(cudaFree(buffer));
    CHECK(cudaFree(U));
    CHECK(cudaFree(s));
    CHECK(cudaFree(S));
    CHECK(cudaFree(VT));

}

int main(int argc, char **argv){

    int n,m,i;

    if(argc != 3){
        printf("usage: cuSolverSVDRebuildTest n m\n");
        exit(-1);
    }

    n = atoi(argv[1]);
    m = atoi(argv[2]);

    if(n < m){
        printf("error: n must be >= m!\n");
        return -1;
    }

    srand(time(NULL));
    double *A,*A2;
    CHECK(cudaMallocHost(&A, n*m*sizeof(double)));
    CHECK(cudaMallocHost(&A2, n*m*sizeof(double)));

    for(i=0; i<n*m; i++)
        A[i] = rand()/(double)RAND_MAX;

    double *d_A,*d_A2;
    CHECK(cudaMalloc(&d_A, n*m*sizeof(double)));
    CHECK(cudaMalloc(&d_A2, n*m*sizeof(double)));

    CHECK(cudaMemcpy(d_A, A, n*m*sizeof(double), cudaMemcpyHostToDevice));

    SVDRebuild(d_A, n, m, d_A2);

    CHECK(cudaMemcpy(A2, d_A2, n*m*sizeof(double), cudaMemcpyDeviceToHost));

    for(i=0; i<n*m; i++)
        if(fabs(A[i] - A2[i]) > 1e-6){
            printf("diff at index %d is: %f\n", i, A[i]-A2[i]);
            return -1;
        }

    printf("OK\n");

}
