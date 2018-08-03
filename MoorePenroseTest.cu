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
    if(CheckPseudoInverse(d_theta, n, m, d_thetaPseudoInv)){
        printf("OK\n");
    }
    else{
        printf("NOPE\n");
    }

    return 0;
}
