#include "vectorUtility.cu"
#include "matrixPrint.h"

/*
Questo programma testa la correttezza del kernel per il calcolo della norma di un vettore.
Stampa il vettore generato randomicamente in un formato facilmente importabile in python per il confronto dei risultati.
*/
int main(int argc, char **argv){

    if(argc != 2){
        printf("usage: normTest <N>\n");
        return 1;
    }
    
    int i,N = atoi(argv[1]);
    double *v, *d_v;

    CHECK(cudaMallocHost(&v, N*sizeof(double)));

    srand(time(NULL));
    for(i=0; i<N; i++)
        v[i] = rand()/(double)RAND_MAX;

    printColumnMajorMatrixForPythonWithPrecision(v,1,N,15);

    CHECK(cudaMalloc(&d_v, N*sizeof(double)));
    CHECK(cudaMemcpy(d_v, v, N*sizeof(double), cudaMemcpyHostToDevice));

    printf("\n%.15f\n",vectorNorm(d_v,N));

}
