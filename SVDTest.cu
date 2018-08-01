#include "SVD.cu"
#include <stdio.h>

int main(int argc, char **argv){

    int n,m,i;

    if(argc != 3){
        printf("usage: SVDTest n m\n");
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

    //Allocate device space for theta and theta2
    float *d_theta, *d_theta2;

    CHECK(cudaMalloc(&d_theta, n*m*sizeof(float)));
    CHECK(cudaMalloc(&d_theta2, n*m*sizeof(float)));
    CHECK(cudaMemcpy(d_theta, theta, n*m*sizeof(float), cudaMemcpyHostToDevice));

    //call SVD
    SVDRebuild(d_theta, n, m, d_theta2);

    //Check result
    float *theta2;
    CHECK(cudaMallocHost(&theta2, n*m*sizeof(float)));
    CHECK(cudaMemcpy(theta2, d_theta2, n*m*sizeof(float), cudaMemcpyDeviceToHost));

    //Check result
    for(i=0; i<n*m; i++)
        if(abs(theta[i] - theta2[i]) > 1e-4)
            break;

    if(i < m){
        printf("NOPE\n");
        printf("%f\n",abs(theta[i] - theta2[i]));
    }
    else
        printf("OK!\n");

    return 0;

}
