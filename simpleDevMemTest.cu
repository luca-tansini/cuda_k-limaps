#include "k-LiMapS.cu"
#include "MoorePenrose.cu"
#include <sys/time.h>

#ifndef _MATRIX_PRINT_H
    #include "matrixPrint.h"
#endif

double seconds(){
	struct timeval tp;
	struct timezone tzp;
	gettimeofday(&tp, &tzp);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

//Kernel implementing the square sum of a vector (vector is destroyed after computation, with v[i] being the partial sum of block i). The exceeding portion of the vector must be set to 0.
__global__ void squareVectorReduceSum(double *v){

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
This main tests the k-LiMapS algorithm reading the dictionary, its pseudoinverse and #iter random k-sparse alpha vectors from stdin.
*/
int main(int argc, char **argv){

    int n,m,k,i,j,numIter;
    scanf("%d", &n);
    scanf("%d", &m);
    scanf("%d", &k);
    scanf("%d", &numIter);

    double *theta, *thetaPseudoInv, *alpha, *limapsAlpha, *d_limapsAlpha;
    CHECK(cudaMallocHost(&theta, n*m*sizeof(double)));
    CHECK(cudaMallocHost(&thetaPseudoInv, n*m*sizeof(double)));
    CHECK(cudaMallocHost(&alpha, m*sizeof(double)));
    CHECK(cudaMallocHost(&limapsAlpha, m*sizeof(double)));
    CHECK(cudaMalloc(&d_limapsAlpha, m*sizeof(double)));

    //Read theta with random values between 0 and 1
    for(i=0; i<n*m; i++)
        scanf("%lf", &theta[i]);
    double *d_theta;
    CHECK(cudaMalloc(&d_theta, n*m*sizeof(double)));
    CHECK(cudaMemcpy(d_theta, theta, n*m*sizeof(double), cudaMemcpyHostToDevice));

    //Read theta Moore-Penrose inverse
    for(i=0; i<m*n; i++)
        scanf("%lf", &thetaPseudoInv[i]);
    double *d_thetaPseudoInv;
    CHECK(cudaMalloc(&d_thetaPseudoInv, m*n*sizeof(double)));
    CHECK(cudaMemcpy(d_thetaPseudoInv, thetaPseudoInv, m*n*sizeof(double), cudaMemcpyHostToDevice));

    //Initialize cublas
    double cualpha=1,cubeta=0;
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    //Allocate device pointers
    double *d_b, *d_alpha;
    CHECK(cudaMalloc(&d_b, n*sizeof(double)));
    CHECK(cudaMalloc(&d_alpha, m*sizeof(double)));

    //Allocate MSE temp pointer
    int blocks = ceil(n*1.0/BLOCK_SIZE);
    dim3 dimGrid(blocks,1,1);
    dim3 dimBlock(BLOCK_SIZE,1,1);
    double *d_limapsB;
    CHECK(cudaMalloc(&d_limapsB, blocks*BLOCK_SIZE*sizeof(double)));
    CHECK(cudaMemset(d_limapsB, 0, blocks*BLOCK_SIZE*sizeof(double)));

    double avgMSE=0;
    double *partialMSEBlocks;
    CHECK(cudaMallocHost(&partialMSEBlocks, blocks*sizeof(double)));
    int succ = 0;
    double t1,avgt=0;

    for(int iter = 0; iter < numIter; iter++ ){

        //Read alpha
        for(i=0; i<m; i++)
            scanf("%lf", &alpha[i]);
        CHECK(cudaMemcpy(d_alpha, alpha, m*sizeof(double), cudaMemcpyHostToDevice));

        //Calculate b = theta * alpha
        CHECK_CUBLAS(cublasDgemv(cublasHandle, CUBLAS_OP_N, n, m, &cualpha, d_theta, n, d_alpha, 1, &cubeta, d_b, 1));

        /*//DEBUG
        double *b;
        CHECK(cudaMallocHost(&b, n*sizeof(double)));
        CHECK(cudaMemcpy(b, d_b, n*sizeof(double), cudaMemcpyDeviceToHost));
        printf("\nb:\n");
        printHighlightedVector(b,n);
        //END DEBUG*/

        //Call k_LiMapS
        t1 = seconds();
        devMemK_LiMapS(k, d_theta, n, m, d_thetaPseudoInv, d_b, d_limapsAlpha, 1000);
        avgt += seconds() - t1;

        /*//DEBUG
        double *limapsAlpha;
        CHECK(cudaMallocHost(&limapsAlpha, m * sizeof(double)));
        CHECK(cudaMemcpy(limapsAlpha, d_limapsAlpha, m*sizeof(double), cudaMemcpyDeviceToHost));
        printf("\nalpha:\n");
        printHighlightedVector(alpha,m);
        printf("\nlimapsAlpha:\n");
        printHighlightedVector(limapsAlpha,m);
        //END DEBUG*/

        //Check result
        CHECK(cudaMemcpy(limapsAlpha, d_limapsAlpha, m*sizeof(double), cudaMemcpyHostToDevice));
        for(i=0; i<m; i++)
            if(abs(alpha[i] - limapsAlpha[i]) > 1e-4)
                break;
        if(i == m)
            succ++;


        //Calculate MSE: sum((b - theta * limapsAlpha)^2)/n
        CHECK_CUBLAS(cublasDgemv(cublasHandle, CUBLAS_OP_N, n, m, &cualpha, d_theta, n, d_limapsAlpha, 1, &cubeta, d_limapsB, 1));

        /*//DEBUG
        double *tmp;
        CHECK(cudaMallocHost(&tmp, blocks*BLOCK_SIZE*sizeof(double)));
        CHECK(cudaMemcpy(tmp, d_limapsB, blocks*BLOCK_SIZE*sizeof(double), cudaMemcpyDeviceToHost));
        printf("\ntheta * limapsAlpha:\n");
        printHighlightedVector(tmp, blocks*BLOCK_SIZE);
        //END DEBUG*/

        vectorSum<<<dimGrid,dimBlock>>>(1, d_b, -1, d_limapsB, d_limapsB, n);
        CHECK(cudaDeviceSynchronize());

        /*//DEBUG
        CHECK(cudaMemcpy(tmp, d_limapsB, blocks*BLOCK_SIZE*sizeof(double), cudaMemcpyDeviceToHost));
        printf("\nb - theta * limapsAlpha:\n");
        printHighlightedVector(tmp, blocks*BLOCK_SIZE);
        //END DEBUG*/

        squareVectorReduceSum<<<dimGrid,dimBlock>>>(d_limapsB);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(partialMSEBlocks, d_limapsB, blocks * sizeof(double), cudaMemcpyDeviceToHost));
        double MSE = 0;
        for(j=0; j<blocks; j++)
            MSE += partialMSEBlocks[j];
        avgMSE += MSE/n;
    }

    avgMSE/=numIter;
    printf("CUDA;%d;%d;%d;%d;%.2f;%.15f;%.6f\n",n,m,k,numIter,succ*100.0/numIter,avgMSE,avgt/numIter);

    //Free memory
    CHECK(cudaFreeHost(theta));
    CHECK(cudaFreeHost(alpha));
    CHECK(cudaFree(d_theta));
    CHECK(cudaFree(d_alpha));
    CHECK(cudaFree(d_thetaPseudoInv));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_limapsB));
    CHECK(cudaFree(d_limapsAlpha));
    cudaDeviceReset();

    return 0;

}
