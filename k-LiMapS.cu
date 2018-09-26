#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <cublas_v2.h>

#ifndef _COMMON_H
    #include "common.h"
#endif

#define BLOCK_SIZE 256

//Compare function required by qsort, descending order
//gives -1 if elem1 > elem2, 1 if elem2 > elem1, 0 if equal
int comp(const void *elem1, const void *elem2) {
    double f = *((double*)elem1);
    double s = *((double*)elem2);
    if (f > s) return -1;
    if (f < s) return 1;
    return 0;
}

//Kernel implementing the F(lambda) shrinkage: b = F(lambda,a)
__global__ void fShrinkage(double lambda, double *a, double *b, int len){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < len){
        b[tid] = a[tid] * (1 - powf(M_E, -lambda*fabsf(a[tid])));
    }
}

//Kernel implementing: res = alpha * a + beta * b
__global__ void vectorSum(double alpha, double *a, double beta, double *b, double *res, int len){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < len)
        res[tid] = alpha * a[tid] + beta * b[tid];
}

//Kernel implementing the final thresholding step
__global__ void thresholding(double *v, int len, double threshold){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < len)
        if(fabsf(v[tid]) < threshold)
            v[tid] = 0;
}

//Kernel implementing the 2-norm of a vector (the vector is destroyed after computation, with v[i] being the partial sum of block i)
__global__ void vector2norm(double *v){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    v[tid] *= v[tid];

    int step = blockDim.x / 2;
    int idx = threadIdx.x;
    double *p = v + blockDim.x * blockIdx.x;
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
void k_LiMapS(int k, double *theta, int n, int m, double *thetaPseudoInv, double *b, double *alpha, int maxIter){

    //Create the cublas handle
    cublasHandle_t handle;
	CHECK_CUBLAS(cublasCreate(&handle));

    //device memory pointers and allocation
    double *d_theta, *d_thetaPseudoInv, *d_b, *d_alpha;

    CHECK(cudaMalloc(&d_theta, n*m*sizeof(double)));
    CHECK(cudaMalloc(&d_thetaPseudoInv, m*n*sizeof(double)));
    CHECK(cudaMalloc(&d_b, n*sizeof(double)));
    CHECK(cudaMalloc(&d_alpha, m*sizeof(double)));

    //transfer dictionary and pseudoinverse to device memory
    CHECK_CUBLAS(cublasSetMatrix(n, m, sizeof(double), theta, n, d_theta, n));
    CHECK_CUBLAS(cublasSetMatrix(m, n, sizeof(double), thetaPseudoInv, m, d_thetaPseudoInv, m));
    CHECK_CUBLAS(cublasSetVector(n, sizeof(double), b, 1, d_b, 1));

    //calculate initial alpha = thetaPseudoInv * b
    double cuAlpha = 1, cuBeta = 0;
    CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_N, m, n, &cuAlpha, d_thetaPseudoInv, m, d_b, 1, &cuBeta, d_alpha, 1));

    //algorithm internal loop
    int i = 0;
    int mBlocks = ceil(m*1.0/BLOCK_SIZE);
    double sigma[m],partialNormBlocks[mBlocks];
    double *d_beta,*d_oldalpha;
    dim3 dimBlock(BLOCK_SIZE,1,1);
    dim3 dimGridM(mBlocks,1,1);
    dim3 dimGridN(ceil(n*1.0/BLOCK_SIZE),1,1);

    CHECK(cudaMalloc(&d_beta, m*sizeof(double)));
    CHECK(cudaMalloc(&d_oldalpha, mBlocks*BLOCK_SIZE*sizeof(double)));
    CHECK(cudaMemset(d_oldalpha, 0, mBlocks*BLOCK_SIZE*sizeof(double)));

    while(i < maxIter){

        //1a. retrieve alpha into sigma
        CHECK_CUBLAS(cublasGetVector(m, sizeof(double), d_alpha, 1, sigma, 1));
        //1b. sort absolute values of sigma in descending order
        for(int j=0; j<m; j++)
            sigma[j] = fabs(sigma[j]);
        qsort(sigma, m, sizeof(double), comp);

        //2. calculate lambda = 1/sigma[k]
        double lambda = 1/sigma[k];

        //3. calculate beta = F(lambda, alpha)
        fShrinkage<<<dimGridM,dimBlock>>>(lambda, d_alpha, d_beta, m);
        CHECK(cudaDeviceSynchronize());

        //4. update alpha = beta - thetaPseudoInv * (theta * beta - b)
        //using aplha for intermediate results (alpha has size m and m >> n)

        //save oldalpha
        CHECK(cudaMemcpy(d_oldalpha, d_alpha, m*sizeof(double), cudaMemcpyDeviceToDevice));

        //alpha = theta * beta (€ R^n)
        CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_N, n, m, &cuAlpha, d_theta, n, d_beta, 1, &cuBeta, d_alpha, 1));

        //alpha = alpha - b (€ R^n)
        vectorSum<<<dimGridN,dimBlock>>>(1, d_alpha, -1, d_b, d_alpha, n);
        CHECK(cudaDeviceSynchronize());

        //alpha = thetaPseudoInv * alpha (€ R^m)
        CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_N, m, n, &cuAlpha, d_thetaPseudoInv, m, d_alpha, 1, &cuBeta, d_alpha, 1));

        //alpha = beta - alpha (€ R^m)
        vectorSum<<<dimGridM,dimBlock>>>(1, d_beta, -1, d_alpha, d_alpha, m);
        CHECK(cudaDeviceSynchronize());

        //loop conditions update
        vectorSum<<<dimGridM,dimBlock>>>(1, d_alpha, -1, d_oldalpha, d_oldalpha, m);
        CHECK(cudaDeviceSynchronize());
        vector2norm<<<dimGridM,dimBlock>>>(d_oldalpha);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(partialNormBlocks, d_oldalpha, mBlocks * sizeof(double), cudaMemcpyDeviceToHost));
        double norm = 0;
        for(int j=0; j<mBlocks; j++)
            norm += partialNormBlocks[j];
        norm = sqrt(norm);
        if(norm < 1e-6)
            break;
        i++;
    }

    //final thresholding step: alpha[i] = 0 if |alpha[i]| <= sigma[k]
    thresholding<<<dimGridM,dimBlock>>>(d_alpha, m, sigma[k]);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(alpha, d_alpha, m*sizeof(double), cudaMemcpyHostToDevice));

    //Free Memory
    CHECK(cudaFree(d_theta));
    CHECK(cudaFree(d_thetaPseudoInv));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_alpha));
    CHECK(cudaFree(d_oldalpha));
    CHECK(cudaFree(d_beta));
    CHECK_CUBLAS(cublasDestroy(handle));

}

/*
Function implementing the k-LiMapS algorithm.
Same as the other one, but parameters are device memory pointers
*/
void devMemK_LiMapS(int k, double *theta, int n, int m, double *thetaPseudoInv, double *b, double *alpha, int maxIter){

    //Create the cublas handle
    cublasHandle_t handle;
	CHECK_CUBLAS(cublasCreate(&handle));

    //calculate initial alpha = thetaPseudoInv * b
    double cuAlpha = 1, cuBeta = 0;
    CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_N, m, n, &cuAlpha, thetaPseudoInv, m, b, 1, &cuBeta, alpha, 1));

    //algorithm internal loop
    int i = 0;
    int mBlocks = ceil(m*1.0/BLOCK_SIZE);
    double sigma[m],partialNormBlocks[mBlocks];
    double *beta,*oldalpha;
    dim3 dimBlock(BLOCK_SIZE,1,1);
    dim3 dimGridM(mBlocks,1,1);
    dim3 dimGridN(ceil(n*1.0/BLOCK_SIZE),1,1);

    CHECK(cudaMalloc(&beta, m*sizeof(double)));
    CHECK(cudaMalloc(&oldalpha, mBlocks*BLOCK_SIZE*sizeof(double)));
    CHECK(cudaMemset(oldalpha, 0, mBlocks*BLOCK_SIZE*sizeof(double)));

    while(i < maxIter){

        //1a. retrieve alpha into sigma
        CHECK_CUBLAS(cublasGetVector(m, sizeof(double), alpha, 1, sigma, 1));
        //1b. sort absolute values of sigma in descending order
        for(int j=0; j<m; j++)
            sigma[j] = fabs(sigma[j]);
        qsort(sigma, m, sizeof(double), comp);

        //2. calculate lambda = 1/sigma[k]
        double lambda = 1/sigma[k];

        //3. calculate beta = F(lambda, alpha)
        fShrinkage<<<dimGridM,dimBlock>>>(lambda, alpha, beta, m);
        CHECK(cudaDeviceSynchronize());

        //4. update alpha = beta - thetaPseudoInv * (theta * beta - b)
        //using aplha for intermediate results (alpha has size m and m >> n)

        //save oldalpha
        CHECK(cudaMemcpy(oldalpha, alpha, m*sizeof(double), cudaMemcpyDeviceToDevice));

        //alpha = theta * beta (€ R^n)
        CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_N, n, m, &cuAlpha, theta, n, beta, 1, &cuBeta, alpha, 1));

        //alpha = alpha - b (€ R^n)
        vectorSum<<<dimGridN,dimBlock>>>(1, alpha, -1, b, alpha, n);
        CHECK(cudaDeviceSynchronize());

        //alpha = thetaPseudoInv * alpha (€ R^m)
        CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_N, m, n, &cuAlpha, thetaPseudoInv, m, alpha, 1, &cuBeta, alpha, 1));

        //alpha = beta - alpha (€ R^m)
        vectorSum<<<dimGridM,dimBlock>>>(1, beta, -1, alpha, alpha, m);
        CHECK(cudaDeviceSynchronize());

        //loop conditions update
        vectorSum<<<dimGridM,dimBlock>>>(1, alpha, -1, oldalpha, oldalpha, m);
        CHECK(cudaDeviceSynchronize());
        vector2norm<<<dimGridM,dimBlock>>>(oldalpha);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(partialNormBlocks, oldalpha, mBlocks * sizeof(double), cudaMemcpyDeviceToHost));
        double norm = 0;
        for(int j=0; j<mBlocks; j++)
            norm += partialNormBlocks[j];
        norm = sqrt(norm);
        if(norm <= 1e-5){
            break;
        }
        i++;
    }

    //final thresholding step: alpha[i] = 0 if |alpha[i]| <= sigma[k]
    thresholding<<<dimGridM,dimBlock>>>(alpha, m, sigma[k]);
    CHECK(cudaDeviceSynchronize());

    //Free Memory
    CHECK(cudaFree(oldalpha));
    CHECK(cudaFree(beta));
    CHECK_CUBLAS(cublasDestroy(handle));

}
