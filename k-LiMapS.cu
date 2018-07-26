#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "cublas_v2.h"
#include "common.h"
#include "matrixPrint.h"

#define BLOCK_SIZE 256

//Compare function required by qsort, descending order
//gives -1 if elem1 > elem2, 1 if elem2 > elem1, 0 if equal
int comp(const void *elem1, const void *elem2) {
    float f = *((float*)elem1);
    float s = *((float*)elem2);
    if (f > s) return -1;
    if (f < s) return 1;
    return 0;
}

//Kernel implementing the F(lambda) shrinkage: b = F(lambda,a)
__global__ void fShrinkage(float lambda, float *a, float *b, int len){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < len){
        b[tid] = a[tid] * (1 - powf(M_E, -lambda*fabsf(a[tid])));
    }
}

//Kernel implementing: res = alpha * a + beta * b
__global__ void vectorSum(float alpha, float *a, float beta, float *b, float *res, int len){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < len)
        res[tid] = alpha * a[tid] + beta * b[tid];
}

//Kernel implementing the final thresholding step
__global__ void thresholding(float *v, int len, float threshold){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < len)
        if(fabsf(v[tid]) < threshold)
            v[tid] = 0;
}

//Kernel implementing the 2-norm of a vector (the vector is destroyed after computation, with v[i] being the partial sum of block i)
__global__ void vector2norm(float *v){

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
    if(idx == 0)
        v[blockIdx.x] = p[idx];
}

/*
Function implementing the k-LiMapS algorithm.
Parameters description:
    k: sparsity level
    theta: the dictionary € R^(n*m)
    thetaPseudoInv: the pseudo inverse of the dictionary (€ R^(m*n))
    b: the signal € R^n
    alpha: the output vector € R^m
    maxIter: a max iteration limit for the internal loop
All matrices are required to be in column-major format for compatibility with the CUBLAS libraries
The result is an aproximate solution for b = theta*alpha s.t. lzero-norm(alpha) <= k
*/
void k_LiMapS(int k, float *theta, int n, int m, float *thetaPseudoInv, float *b, float *alpha, int maxIter){

    //Create the cublas handle
    cublasHandle_t handle;
	CHECK_CUBLAS(cublasCreate(&handle));

    //device memory pointers and allocation
    float *d_theta, *d_thetaPseudoInv, *d_b, *d_alpha;

    CHECK(cudaMalloc(&d_theta, n*m*sizeof(float)));
    CHECK(cudaMalloc(&d_thetaPseudoInv, m*n*sizeof(float)));
    CHECK(cudaMalloc(&d_b, n*sizeof(float)));
    CHECK(cudaMalloc(&d_alpha, m*sizeof(float)));

    //transfer dictionary and pseudoinverse to device memory
    CHECK_CUBLAS(cublasSetMatrix(n, m, sizeof(float), theta, n, d_theta, n));
    CHECK_CUBLAS(cublasSetMatrix(m, n, sizeof(float), thetaPseudoInv, m, d_thetaPseudoInv, m));
    CHECK_CUBLAS(cublasSetVector(n, sizeof(float), b, 1, d_b, 1));
    CHECK(cudaMemset(d_alpha, 0, m*sizeof(float)));

    //calculate initial alpha = thetaPseudoInv * b
    float cuAlpha = 1, cuBeta = 0;
    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, m, n, &cuAlpha, d_thetaPseudoInv, m, d_b, 1, &cuBeta, d_alpha, 1));

    //algorithm internal loop
    int i = 0;
    int mBlocks = ceil(m*1.0/BLOCK_SIZE);
    float sigma[m],partialNormBlocks[mBlocks];
    float *d_beta,*d_oldalpha;
    dim3 dimBlock(BLOCK_SIZE,1,1);
    dim3 dimGridM(mBlocks,1,1);
    dim3 dimGridN(ceil(n*1.0/BLOCK_SIZE),1,1);

    CHECK(cudaMalloc(&d_beta, m*sizeof(float)));
    CHECK(cudaMalloc(&d_oldalpha, mBlocks*BLOCK_SIZE*sizeof(float)));
    CHECK(cudaMemset(d_oldalpha, 0, mBlocks*BLOCK_SIZE*sizeof(float)));

    /*//DEBUG ALLOCATIONS
    float *beta;

    CHECK(cudaMallocHost(&beta, m*sizeof(float)));
    //END DEBUG*/

    while(i < maxIter){

        //1a. retrieve alpha into sigma
        CHECK_CUBLAS(cublasGetVector(m, sizeof(float), d_alpha, 1, sigma, 1));
        //1b. sort absolute values of sigma in descending order
        for(int j=0; j<m; j++)
            sigma[j] = abs(sigma[j]);
        qsort(sigma, m, sizeof(float), comp);

        /*//DEBUG SIGMA
        printf("\nIter #%d:\n",i);
        printf("sigma:\n");
        printColumnMajorMatrix(sigma,1,m);
        //END DEBUG*/

        //2. calculate lambda = 1/sigma[k]
        float lambda = 1/sigma[k];

        //3. calculate beta = F(lambda, alpha)
        fShrinkage<<<dimGridM,dimBlock>>>(lambda, d_alpha, d_beta, m);
        CHECK(cudaDeviceSynchronize());

        /*//DEBUG BETA
        CHECK(cudaMemcpy(beta, d_beta, m*sizeof(float), cudaMemcpyDeviceToHost));
        printf("beta:\n");
        printColumnMajorMatrix(beta,1,m);
        //END DEBUG*/

        //4. update alpha = beta - thetaPseudoInv * (theta * beta - b)
        //using aplha for intermediate results (alpha has size m and m >> n)

        //save oldalpha
        CHECK(cudaMemcpy(d_oldalpha, d_alpha, m*sizeof(float), cudaMemcpyDeviceToDevice));

        //alpha = theta * beta (€ R^n)
        CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, n, m, &cuAlpha, d_theta, n, d_beta, 1, &cuBeta, d_alpha, 1));

        //alpha = alpha - b (€ R^n)
        vectorSum<<<dimGridN,dimBlock>>>(1, d_alpha, -1, d_b, d_alpha, n);
        CHECK(cudaDeviceSynchronize());

        //alpha = thetaPseudoInv * alpha (€ R^m)
        CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, m, n, &cuAlpha, d_thetaPseudoInv, m, d_alpha, 1, &cuBeta, d_alpha, 1));

        //alpha = beta - alpha (€ R^m)
        vectorSum<<<dimGridM,dimBlock>>>(1, d_beta, -1, d_alpha, d_alpha, m);
        CHECK(cudaDeviceSynchronize());

        //loop conditions update
        vectorSum<<<dimGridM,dimBlock>>>(1, d_alpha, -1, d_oldalpha, d_oldalpha, m);
        CHECK(cudaDeviceSynchronize());
        vector2norm<<<dimGridM,dimBlock>>>(d_oldalpha);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(partialNormBlocks, d_oldalpha, mBlocks * sizeof(float), cudaMemcpyDeviceToHost));
        float norm = 0;
        for(int j=0; j<mBlocks; j++)
            norm += partialNormBlocks[j];
        norm = sqrt(norm);
        /*//DEBUG NORM
        printf("norm:\n%f\n",norm);
        sleep(1);
        //END DEBUG*/
        if(norm < 1e-6){
            //printf("\niter #%d\n", i);
            break;
        }
        i++;
    }

    //final thresholding step: alpha[i] = 0 if |alpha[i]| <= sigma[k]
    thresholding<<<dimGridM,dimBlock>>>(d_alpha, m, sigma[k]);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(alpha, d_alpha, m*sizeof(float), cudaMemcpyHostToDevice));

    //Free memory

}
