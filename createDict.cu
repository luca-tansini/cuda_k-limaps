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

/*
Function generating the dictionary
The values are extracted from a normal distribution (mean 0, stdev 1.0)
Then each column is normalized, dividing each element by the column's norm
*/
void createDict(double *D, int n, int m){

    int blocksperdict = ceil(n*m*1.0/BLOCK_SIZE);
    int blockspercol = ceil(n*1.0/BLOCK_SIZE);

    srand(time(NULL));
    int seed = rand();

    curandState *devStates;
    CHECK(cudaMalloc((void **)&devStates, blocksperdict*BLOCK_SIZE*sizeof(curandState)));

    normfill<<<blocksperdict,BLOCK_SIZE>>>(D, n*m, devStates, seed);
    CHECK(cudaDeviceSynchronize());

    double *tmpcol;
    CHECK(cudaMalloc(&tmpcol, blockspercol*BLOCK_SIZE*sizeof(double)));

    for(int i=0; i<m; i++){

        //use a copy because norm computation destroys the vector
        CHECK(cudaMemcpy(tmpcol, &D[i*n], n*sizeof(double), cudaMemcpyDeviceToDevice));

        //CALCOLA NORMA
        double norm = vectorNorm(tmpcol,n);

        //CHIAMA KERNEL CHE DIVIDE OGNI ELEMENTO PER LA NORMA
        divide<<<blockspercol,BLOCK_SIZE>>>(&D[i*n], norm, n);
        CHECK(cudaDeviceSynchronize());
    }
}

/*
Function generating the k-sparse vector alpha
The k values are extracted from a normal distribution (mean 0, stdev 1.0)
*/
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
