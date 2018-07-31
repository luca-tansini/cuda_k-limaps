#include "MoorePenrose.cu"
#include <stdio.h>

/*
This main tests the MoorePenrose pseudoinverse algorithm reading all the data needed for the test from standard input. Since our matrices have n < m, we transpose theta before calling MoorePenrose. This will give us the pseudoinverse in a transposed state and we only have to transpose it again to obtain the correct pseudoinverse.
*/
int main(int argc, char **argv){

    int n,k,m,i;
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));


    scanf("%d",&n);
    scanf("%d",&k);
    m = n*k;

    //Read dictionary theta
    float *theta;
    CHECK(cudaMallocHost(&theta, n*m*sizeof(float)));
    for(i=0; i<n*m; i++)
        scanf("%f",theta+i);

    //Allocate space for theta and thetaPseudoInv
    float *d_theta, d_thetaPseudoInv;

    CHECK(cudaMalloc(&d_theta, n*m*sizeof(float)));
    CHECK(cudaMalloc(&d_thetaPseudoInv, n*m*sizeof(float)));
    CHECK(cudaMemcpy(d_theta, theta, n*m*sizeof(float)));

    //transpose theta
    CHECK_CUBLAS(cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, d_theta, n, &beta, d_theta, n, d_theta, n));

    //call MoorePenrose
    MoorePenroseInverse(theta, m, n, thetaPseudoInv);

    //transpose result
    CHECK_CUBLAS(cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, d_thetaPseudoInv, n, &beta, d_thetaPseudoInv, n, d_thetaPseudoInv, n));

    //Check result
    float *thetaPseudoInv;
    CHECK(cudaMallocHost(&thetaPseudoInv, m*n*sizeof(float)));
    CHECK(cudaMemcpy(thetaPseudoInv, d_thetaPseudoInv, m*n*sizeof(float), cudaMemcpyDeviceToHost));

    printColumnMajorMatrix(thetaPseudoInv, m, n);

    return 0;

}
