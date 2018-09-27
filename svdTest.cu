#include <stdlib.h>
#include <unistd.h>
#include "MoorePenrose.cu"
#include <stdio.h>

double seconds(){
	struct timeval tp;
	struct timezone tzp;
	gettimeofday(&tp, &tzp);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

int main(int argc, char **argv){

    int n,m,i;

    if(argc != 3){
        printf("usage: svdTest n m\n");
        exit(-1);
    }

    n = atoi(argv[1]);
    m = atoi(argv[2]);

    srand(time(NULL));
    double *A;
    A = (double *) malloc(n*m*sizeof(double));
    for(i=0; i<n*m; i++)
        A[i] = rand()/(double)RAND_MAX;

    double *Apinv = (double *) malloc(m*n*sizeof(double));

    //device memory transfer, useless step but it's just for testing
    double *d_A, *d_Apinv;
    CHECK(cudaMalloc(&d_A, n*m*sizeof(double)));
    CHECK(cudaMalloc(&d_Apinv, m*n*sizeof(double)));
    CHECK(cudaMemcpy(d_A, A, n*m*sizeof(double), cudaMemcpyHostToDevice));

    double t = seconds();
    HostMoorePenroseInverse(d_A, n, m, d_Apinv);
    printf("HostMoorePenroseInverse computation time: %f\n",seconds() - t);

    if(!CheckPseudoInverse(d_A, n, m, d_Apinv))
        printf("Something went wrong with the Moore-Penrose pseudoinverse!\n");
    else
        printf("OK\n");

    return 0;
}
