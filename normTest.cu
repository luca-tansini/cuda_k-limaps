#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "common.h"
#include "matrixPrint.h"

#define BLOCK_SIZE 256

__global__ void vector2norm(float *v, int len){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    v[tid] *= v[tid];

    int step = blockDim.x / 2;
    int idx = threadIdx.x;
    float *p = v + blockDim.x * blockIdx.x;
    while(step > 0){
        if(idx < step)
            p[idx] = p[idx] + p[idx+step];
        step /= 2;
        __syncthreads();
    }
    if(idx == 0){
        printf("%d : %.5f\n",tid,p[idx]);
        v[blockIdx.x] = p[idx];
    }
}

int main(){

    int m = 1000;
    float *v;
    CHECK(cudaMallocHost(&v, m*sizeof(float)));

    int blocks = ceil(m*1.0/BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE,1,1);
    dim3 dimGrid(blocks,1,1);

    srand(time(NULL));
    for(int i=0; i<m; i++)
        v[i] = rand()/(float)RAND_MAX;

    float *d_v;
    CHECK(cudaMalloc(&d_v, blocks*BLOCK_SIZE*sizeof(float)));
    CHECK(cudaMemset(d_v, 0, blocks*BLOCK_SIZE*sizeof(float)));
    CHECK(cudaMemcpy(d_v , v, m*sizeof(float), cudaMemcpyHostToDevice));

    printColumnMajorMatrixForPython(v, 1, m);

    vector2norm<<<dimGrid,dimBlock>>>(d_v, blocks*BLOCK_SIZE);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(v, d_v, blocks * sizeof(float), cudaMemcpyDeviceToHost));

    float norm = 0;
    for(int i=0; i<blocks; i++)
        norm += v[i];

    norm = sqrt(norm);

    printf("\nnorm: %.5f\n", norm);
    CHECK(cudaFreeHost(v));
    CHECK(cudaFree(d_v));
    cudaDeviceReset();

}
