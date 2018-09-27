#include "MoorePenrose.cu"
#include <stdio.h>

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
    double *theta;
    CHECK(cudaMallocHost(&theta, n*m*sizeof(double)));
    for(i=0; i<n*m; i++)
        theta[i] = rand()/(double)RAND_MAX;

    //Allocate space for theta and thetaPseudoInv
    double *d_theta, *d_thetaPseudoInv;

    CHECK(cudaMalloc(&d_theta, n*m*sizeof(double)));
    CHECK(cudaMalloc(&d_thetaPseudoInv, m*n*sizeof(double)));
    CHECK(cudaMemcpy(d_theta, theta, n*m*sizeof(double), cudaMemcpyHostToDevice));

    //call MoorePenrose
    MoorePenroseInverse(d_theta, n, m, d_thetaPseudoInv);

    //Check result
    if(CheckPseudoInverse(d_theta, n, m, d_thetaPseudoInv)){
        printf("OK\n");
    }
    else{
        printf("NOPE\n");
    }

    return 0;
}
