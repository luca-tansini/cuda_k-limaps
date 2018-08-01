#include "MoorePenrose.cu"
#include <stdio.h>

/*
This main tests the MoorePenrose pseudoinverse algorithm reading all the data needed for the test from standard input. Since our matrices have n < m, we transpose theta before calling MoorePenrose. This will give us the pseudoinverse in a transposed state and we only have to transpose it again to obtain the correct pseudoinverse.
*/
int main(int argc, char **argv){

    int n,m,i;

    if(argc != 3){
        printf("usage: MoorePenroseTest n m\n");
        exit(-1);
    }

    n = atoi(argv[1]);
    m = atoi(argv[2]);

    //Generate dictionary theta
    srand(time(NULL));
    float *theta;
    CHECK(cudaMallocHost(&theta, n*m*sizeof(float)));
    for(i=0; i<n*m; i++)
        theta[i] = rand()/(float)RAND_MAX;

    //Allocate space for theta and thetaPseudoInv
    float *d_theta, *d_thetaPseudoInv;

    CHECK(cudaMalloc(&d_theta, n*m*sizeof(float)));
    CHECK(cudaMalloc(&d_thetaPseudoInv, m*n*sizeof(float)));
    CHECK(cudaMemcpy(d_theta, theta, n*m*sizeof(float), cudaMemcpyHostToDevice));

    //call MoorePenrose
    MoorePenroseInverse(d_theta, n, m, d_thetaPseudoInv);

    //Check result
    float *thetaPseudoInv;
    CHECK(cudaMallocHost(&thetaPseudoInv, m*n*sizeof(float)));
    CHECK(cudaMemcpy(thetaPseudoInv, d_thetaPseudoInv, m*n*sizeof(float), cudaMemcpyDeviceToHost));
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    float *id,*d_id,alpha=1,beta=0;
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
        printf("\nthetaPseudoInv:\n");
        printColumnMajorMatrix(thetaPseudoInv, m, n);
        printf("\nthetaPseudoInv * theta:\n");
        printColumnMajorMatrix(id, m, m);
    }
    else
        printf("OK!\n");

    return 0;
}
