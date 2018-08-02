#include "MoorePenrose.cu"
#include <stdio.h>

/*
This main tests the workaround for computing the pseudo inverse of a [n x m] matrix with n < m. The workaround is to transpose the matrix, calculate its pseudoinverse and then transpose the obtained pseudoinverse
*/
int main(int argc, char **argv){

    int n,m,i;

    if(argc != 3){
        printf("usage: MoorePenroseTransposeTest n m\n");
        exit(-1);
    }

    n = atoi(argv[1]);
    m = atoi(argv[2]);

    //Initialize cublas
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    float alpha, beta;
    alpha = 1;
    beta = 0;

    //Generate dictionary theta
    srand(time(NULL));
    float *theta;
    CHECK(cudaMallocHost(&theta, n*m*sizeof(float)));
    for(i=0; i<n*m; i++)
        theta[i] = rand()/(float)RAND_MAX;

    //Allocate space for theta and thetaPseudoInv
    float *d_theta, *d_thetaT, *d_thetaPseudoInvT, *d_thetaPseudoInv;

    CHECK(cudaMalloc(&d_theta, n*m*sizeof(float)));
    CHECK(cudaMalloc(&d_thetaT, m*n*sizeof(float)));
    CHECK(cudaMalloc(&d_thetaPseudoInvT, n*m*sizeof(float)));
    CHECK(cudaMalloc(&d_thetaPseudoInv, m*n*sizeof(float)));
    CHECK(cudaMemcpy(d_theta, theta, n*m*sizeof(float), cudaMemcpyHostToDevice));

    //Transpose theta
    CHECK_CUBLAS(cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, d_theta, n, &beta, d_theta, n, d_thetaT, m));

    //call MoorePenrose
    MoorePenroseInverse(d_theta, m, n, d_thetaPseudoInvT);

    //Transpose thetaPseudoInv
    CHECK_CUBLAS(cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, d_thetaPseudoInvT, n, &beta, d_thetaPseudoInvT, n, d_thetaPseudoInv, m));

    //Check result
    float *id,*d_id;
    CHECK(cudaMalloc(&d_id, m*m*sizeof(float)));

    CHECK_CUBLAS(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, m, n, &alpha, d_thetaPseudoInv, m, d_theta, n, &beta, d_id, m));

    CHECK(cudaMallocHost(&id, m*m*sizeof(float)));
    CHECK(cudaMemcpy(id, d_id, m*m*sizeof(float), cudaMemcpyDeviceToHost));

    for(i=0;i<m;i++)
        for(int j=0; j<m; j++){
            if(i == j && abs(1-id[j*m+i]) > 1e-4) break;
            if(i != j && abs(0-id[j*m+i]) > 1e-4) break;
        }

    if(i < m){
        printf("NOPE!\n");
        printf("\ntheta:\n");
        printColumnMajorMatrixForPython(theta, n, m);

        float *thetaPseudoInv;
        CHECK(cudaMallocHost(&thetaPseudoInv, m*n*sizeof(float)));
        CHECK(cudaMemcpy(thetaPseudoInv, d_thetaPseudoInv, m*n*sizeof(float), cudaMemcpyDeviceToHost));
        printf("\nthetaPseudoInv:\n");
        printColumnMajorMatrix(thetaPseudoInv, m, n);

        printf("\nthetaPseudoInv * theta:\n");
        printColumnMajorMatrix(id, m, m);
    }
    else
        printf("OK!\n");

    return 0;
}
