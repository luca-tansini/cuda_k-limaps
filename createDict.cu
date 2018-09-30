#include <curand.h>
#include <curand_kernel.h>
#include "vectorUtility.cu"

#define BLOCK_SIZE 256

__global__ void normfill(double *D, int len, curandState *states, int seed){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < len){
        curand_init(tid*seed+seed, 0, 0, &states[tid]);
        D[tid] = curand_uniform_double(&states[tid]);
    }

}

__global__ void divide(double *v, double x, int len){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < len)
        v[tid] /= x;

}

//Function generating the dictionary
//The values are extracted from a normal distribution (mean 0, stdev 1.0)
//Then each column is normalized, dividing each element by the column's norm
void createDict(double *D, int n, int m){

    int blocksperdict = ceil(n*m*1.0/BLOCK_SIZE);

    srand(time(NULL));
    int seed = rand();

    curandState *devStates;
    CHECK(cudaMalloc((void **)&devStates, blocksperdict*BLOCK_SIZE*sizeof(curandState)));

    normfill<<<blocksperdict,BLOCK_SIZE>>>(D, n*m, devStates, seed);
    CHECK(cudaDeviceSynchronize());

    double *tmpcol,*partialNormBlocks,norm;
    int blockspercol = ceil(n*1.0/BLOCK_SIZE);
    CHECK(cudaMalloc(&tmpcol, blockspercol*BLOCK_SIZE*sizeof(double)));
    CHECK(cudaMemset(tmpcol, 0, blockspercol*BLOCK_SIZE*sizeof(double)));
    CHECK(cudaMallocHost(&partialNormBlocks, blockspercol*sizeof(double)));

    for(int i=0; i<m; i++){

        CHECK(cudaMemcpy(tmpcol, &D[i*n], n*sizeof(double), cudaMemcpyDeviceToDevice));

        //CALCOLA NORMA CON vector2norm
        vector2norm<<<blockspercol,BLOCK_SIZE>>>(tmpcol);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(partialNormBlocks, tmpcol, blockspercol*sizeof(double), cudaMemcpyDeviceToHost));
        norm = 0;
        for(int j=0; j<blockspercol; j++)
            norm += partialNormBlocks[j];
        norm = sqrt(norm);

        //CHIAMA KERNEL CHE DIVIDE OGNI ELEMENTO PER LA NORMA
        divide<<<blockspercol,BLOCK_SIZE>>>(&D[i*n], norm, n);
        CHECK(cudaDeviceSynchronize());
    }
}

void generateAlpha(double *alpha, int m, int k){

    int blocksperk = ceil(k*1.0/BLOCK_SIZE);

    srand(time(NULL));
    int seed = rand();

    curandState *devStates;
    CHECK(cudaMalloc(&devStates, blocksperk*BLOCK_SIZE*sizeof(curandState)));

    double *d_kvalues,h_kvalues[k];
    CHECK(cudaMalloc(&d_kvalues, k*sizeof(double)));

    normfill<<<blocksperk,BLOCK_SIZE>>>(d_kvalues, k, devStates, seed);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_kvalues, d_kvalues, k*sizeof(double), cudaMemcpyDeviceToHost));

    double h_alpha[m];
    memset(h_alpha, 0, m*sizeof(double));
    for(int i=0; i<k; i++){
        int idx = rand()%m;
        if(h_alpha[idx] != 0)
            i--;
        else
            h_alpha[idx] = h_kvalues[i];
    }

    CHECK(cudaMemcpy(alpha, h_alpha, m * sizeof(double), cudaMemcpyHostToDevice));

}
