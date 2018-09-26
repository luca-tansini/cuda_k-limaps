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

    int iter,succ=0;
    double *theta,*d_theta, *d_thetaPseudoInv;
    CHECK(cudaMallocHost(&theta, n*m*sizeof(double)));
    CHECK(cudaMalloc(&d_theta, n*m*sizeof(double)));
    CHECK(cudaMalloc(&d_thetaPseudoInv, m*n*sizeof(double)));

    for(iter = 0; iter < 100; iter++){

        //Generate dictionary theta
        for(i=0; i<n*m; i++)
            theta[i] = rand()/(double)RAND_MAX;

        CHECK(cudaMemcpy(d_theta, theta, n*m*sizeof(double), cudaMemcpyHostToDevice));

        //call TransposedMoorePenroseInverse
        TransposedMoorePenroseInverse(d_theta, n, m, d_thetaPseudoInv);

        //Check result
        if(CheckPseudoInverse(d_theta, n, m, d_thetaPseudoInv)){
            //printf("OK\n");
            succ++;
        }
        else{
            printf("NOPE!\n");
            /*
            printf("D:\n");
            printColumnMajorMatrixForPython(theta, n, m);
            printf("\n");

            TransposeDebugMoorePenroseInverse(d_theta, n, m, d_thetaPseudoInv);

            double *h_thetaPseudoInv;
            CHECK(cudaMallocHost(&h_thetaPseudoInv, m*n*sizeof(double)));
            CHECK(cudaMemcpy(h_thetaPseudoInv, d_thetaPseudoInv, m*n*sizeof(double), cudaMemcpyDeviceToHost));

            printf("DINV:\n");
            printColumnMajorMatrixForPython(h_thetaPseudoInv, m, n);
            */
        }
    }

    printf("success: %d%%\n",succ);

    return 0;
}
