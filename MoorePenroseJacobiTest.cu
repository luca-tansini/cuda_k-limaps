#include "MoorePenrose.cu"
#include <stdio.h>

/*
Questo programma testa l'algoritmo per il calcolo della pseudoinversa di MoorePenrose che utilizza il metodo di Jacobi.
*/
int main(int argc, char **argv){

    int n,m,i;

    if(argc != 3){
        printf("usage: JacobiMoorePenroseTest n m\n");
        exit(-1);
    }

    n = atoi(argv[1]);
    m = atoi(argv[2]);

    srand(time(NULL));
    double *A;
    CHECK(cudaMallocHost(&A, n*m*sizeof(double)));

    for(i=0; i<n*m; i++)
        A[i] = rand()/(double)RAND_MAX;

    double *d_A,*d_Apinv;
    CHECK(cudaMalloc(&d_A, n*m*sizeof(double)));
    CHECK(cudaMalloc(&d_Apinv, n*m*sizeof(double)));

    CHECK(cudaMemcpy(d_A, A, n*m*sizeof(double), cudaMemcpyHostToDevice));

    //Chiama JacobiMoorePenroseInverse
    double t = seconds();
    JacobiMoorePenroseInverse(d_A, n, m, d_Apinv);
    printf("elapsed time: %fs\n",seconds() - t);

    //Controllo del risultato
    if(CheckPseudoinverse(d_A, n, m, d_Apinv)){
        printf("OK\n");
    }
    else{
        printf("NOPE\n");
    }

    return 0;
}
