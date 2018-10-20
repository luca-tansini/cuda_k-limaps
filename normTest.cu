#include "vectorUtility.cu"
#include "matrixPrint.h"

int main(int argc, char **argv){

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
