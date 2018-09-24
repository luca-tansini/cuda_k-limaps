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

    if(n > m){
        printf("n must be <= m!\n");
        return -1;
    }

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

    //call TransposedMoorePenroseInverse
    TransposedMoorePenroseInverse(d_theta, n, m, d_thetaPseudoInv);

    //Check result
    if(CheckPseudoInverse(d_theta, n, m, d_thetaPseudoInv)){
        printf("OK\n");
    }
    else{

        printf("D:\n");
        printColumnMajorMatrixForPython(theta, n, m);
        printf("\n");

        TransposeDebugMoorePenroseInverse(d_theta, n, m, d_thetaPseudoInv);

        /*float *h_thetaPseudoInv;
        CHECK(cudaMallocHost(&h_thetaPseudoInv, m*n*sizeof(float)));
        CHECK(cudaMemcpy(h_thetaPseudoInv, d_thetaPseudoInv, m*n*sizeof(float), cudaMemcpyDeviceToHost));

        printf("DINV:\n");
        printColumnMajorMatrixForPython(h_thetaPseudoInv, m, n);
*/
    }

    return 0;
}
