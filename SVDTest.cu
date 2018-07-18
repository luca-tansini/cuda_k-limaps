#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cublas_v2.h"
#include "common.h"
#include "matrixPrint.h"
#include <cusolverDn.h>

int main(int argc, char **argv){

    int n,m,i;
    n = 7;
    m = 3;

    srand(time(NULL));

    float theta[n*m];
    //Fill theta with random values between 0 and 1
    for(i=0; i<n*m; i++)
        theta[i] = rand()/(float)RAND_MAX;

    //DEBUG
    printf("theta:\n");
    printColumnMajorMatrix(theta, n, m);

    printf("theta for Python use:\n");
    printColumnMajorMatrixForPython(theta, n, m);
    //END DEBUG

    float *d_theta,*d_thetaPseudoInv;
    CHECK(cudaMalloc(&d_theta, n*m*sizeof(float)));
    CHECK(cudaMemcpy(d_theta, theta, n*m*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_thetaPseudoInv, m*n*sizeof(float)));

    cusolverDnHandle_t cusolverHandle;
    cublasHandle_t cublasHandle;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverHandle));
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    //Get dimension needed for the workspace buffer and allocate it
    int bufferDim;
    CHECK_CUSOLVER(cusolverDnSgesvd_bufferSize(cusolverHandle, n, m, &bufferDim));
    float *buffer;
    CHECK(cudaMalloc(&buffer,bufferDim));

    //Allocate U,S,V_T
    float *U,*S,*V_T;
    CHECK(cudaMalloc(&U, n*n*sizeof(float)));
    //S that should be a diagonal matrix is returned as a simple vector instead
    CHECK(cudaMalloc(&S, m*sizeof(float)));
    CHECK(cudaMalloc(&V_T, m*m*sizeof(float)));

    //Calculate SVD
    int *dev_info, h_dev_info;
    CHECK(cudaMalloc(&dev_info, sizeof(int)));
    CHECK_CUSOLVER(cusolverDnSgesvd(cusolverHandle, 'A', 'A', n, m, d_theta, n, S, U, n, V_T, m, buffer, bufferDim, NULL, dev_info));

    CHECK(cudaMemcpy(&h_dev_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost));

    //retrieve results
    float *h_U, *h_S, *h_V_T;

    CHECK(cudaMallocHost(&h_U,n*n*sizeof(float)));
    CHECK(cudaMallocHost(&h_S,m*sizeof(float)));
    CHECK(cudaMallocHost(&h_V_T,m*m*sizeof(float)));

    CHECK(cudaMemcpy(h_U, U, n*n*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_S, S, m*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_V_T, V_T, m*m*sizeof(float), cudaMemcpyDeviceToHost));

    printf("\n\nU (%d*%d):\n\n",n,n);
    printColumnMajorMatrix(h_U,n,n);

    printf("\n\nS (%d):\n\n",m);
    for(i=0;i<m;i++)
        printf("%.3f ", h_S[i]);

    printf("\n\nV_T (%d*%d):\n\n",m,m);
    printColumnMajorMatrix(h_V_T,m,m);

    printf("\n");

    return 0;

}
