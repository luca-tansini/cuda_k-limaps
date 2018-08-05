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
__global__ void squareVectorReduceSum(float *v){

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
This main tests the k-LiMapS algorithm reading the dictionary, its pseudoinverse and #iter random k-sparse alpha vectors from stdin.
*/
int main(int argc, char **argv){

    int n,m,k,i,j,numIter;
    scanf("%d", &n);
    scanf("%d", &m);
    scanf("%d", &k);
    scanf("%d", &numIter);

    float *theta, *thetaPseudoInv, *alpha, *limapsAlpha, *d_limapsAlpha;
    CHECK(cudaMallocHost(&theta, n*m*sizeof(float)));
    CHECK(cudaMallocHost(&thetaPseudoInv, n*m*sizeof(float)));
    CHECK(cudaMallocHost(&alpha, m*sizeof(float)));
    CHECK(cudaMallocHost(&limapsAlpha, m*sizeof(float)));
    CHECK(cudaMalloc(&d_limapsAlpha, m*sizeof(float)));

    //Read theta with random values between 0 and 1
    for(i=0; i<n*m; i++)
        scanf("%f", &theta[i]);
    float *d_theta;
    CHECK(cudaMalloc(&d_theta, n*m*sizeof(float)));
    CHECK(cudaMemcpy(d_theta, theta, n*m*sizeof(float), cudaMemcpyHostToDevice));

    //Read theta Moore-Penrose inverse
    for(i=0; i<m*n; i++)
        scanf("%f", &thetaPseudoInv[i]);
    float *d_thetaPseudoInv;
    CHECK(cudaMalloc(&d_thetaPseudoInv, m*n*sizeof(float)));
    CHECK(cudaMemcpy(d_thetaPseudoInv, thetaPseudoInv, m*n*sizeof(float), cudaMemcpyHostToDevice));

    //Initialize cublas
    float cualpha=1,cubeta=0;
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    //Allocate device pointers
    float *d_b, *d_alpha;
    CHECK(cudaMalloc(&d_b, n*sizeof(float)));
    CHECK(cudaMalloc(&d_alpha, m*sizeof(float)));

    //Allocate MSE temp pointer
    int blocks = ceil(n*1.0/BLOCK_SIZE);
    dim3 dimGrid(blocks,1,1);
    dim3 dimBlock(BLOCK_SIZE,1,1);
    float *d_limapsB;
    CHECK(cudaMalloc(&d_limapsB, blocks*BLOCK_SIZE*sizeof(float)));
    CHECK(cudaMemset(d_limapsB, 0, blocks*BLOCK_SIZE*sizeof(float)));

    float avgMSE=0;
    float *partialMSEBlocks;
    CHECK(cudaMallocHost(&partialMSEBlocks, blocks*sizeof(float)));
    int succ = 0;
    double t1,avgt=0;

    for(int iter = 0; iter < numIter; iter++ ){

        //Read alpha
        for(i=0; i<m; i++)
            scanf("%f", &alpha[i]);
        CHECK(cudaMemcpy(d_alpha, alpha, m*sizeof(float), cudaMemcpyHostToDevice));

        //Calculate b = theta * alpha
        CHECK_CUBLAS(cublasSgemv(cublasHandle, CUBLAS_OP_N, n, m, &cualpha, d_theta, n, d_alpha, 1, &cubeta, d_b, 1));

        /*//DEBUG
        float *b;
        CHECK(cudaMallocHost(&b, n*sizeof(float)));
        CHECK(cudaMemcpy(b, d_b, n*sizeof(float), cudaMemcpyDeviceToHost));
        printf("\nb:\n");
        printHighlightedVector(b,n);
        //END DEBUG*/

        //Call k_LiMapS
        t1 = seconds();
        devMemK_LiMapS(k, d_theta, n, m, d_thetaPseudoInv, d_b, d_limapsAlpha, 1000);
        avgt += seconds() - t1;

        /*//DEBUG
        float *limapsAlpha;
        CHECK(cudaMallocHost(&limapsAlpha, m * sizeof(float)));
        CHECK(cudaMemcpy(limapsAlpha, d_limapsAlpha, m*sizeof(float), cudaMemcpyDeviceToHost));
        printf("\nalpha:\n");
        printHighlightedVector(alpha,m);
        printf("\nlimapsAlpha:\n");
        printHighlightedVector(limapsAlpha,m);
        //END DEBUG*/

        //Check result
        CHECK(cudaMemcpy(limapsAlpha, d_limapsAlpha, m*sizeof(float), cudaMemcpyHostToDevice));
        for(i=0; i<m; i++)
            if(abs(alpha[i] - limapsAlpha[i]) > 1e-4)
                break;
        if(i == m)
            succ++;


        //Calculate MSE: sum((b - theta * limapsAlpha)^2)/n
        CHECK_CUBLAS(cublasSgemv(cublasHandle, CUBLAS_OP_N, n, m, &cualpha, d_theta, n, d_limapsAlpha, 1, &cubeta, d_limapsB, 1));

        /*//DEBUG
        float *tmp;
        CHECK(cudaMallocHost(&tmp, blocks*BLOCK_SIZE*sizeof(float)));
        CHECK(cudaMemcpy(tmp, d_limapsB, blocks*BLOCK_SIZE*sizeof(float), cudaMemcpyDeviceToHost));
        printf("\ntheta * limapsAlpha:\n");
        printHighlightedVector(tmp, blocks*BLOCK_SIZE);
        //END DEBUG*/

        vectorSum<<<dimGrid,dimBlock>>>(1, d_b, -1, d_limapsB, d_limapsB, n);
        CHECK(cudaDeviceSynchronize());

        /*//DEBUG
        CHECK(cudaMemcpy(tmp, d_limapsB, blocks*BLOCK_SIZE*sizeof(float), cudaMemcpyDeviceToHost));
        printf("\nb - theta * limapsAlpha:\n");
        printHighlightedVector(tmp, blocks*BLOCK_SIZE);
        //END DEBUG*/

        squareVectorReduceSum<<<dimGrid,dimBlock>>>(d_limapsB);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(partialMSEBlocks, d_limapsB, blocks * sizeof(float), cudaMemcpyDeviceToHost));
        float MSE = 0;
        for(j=0; j<blocks; j++)
            MSE += partialMSEBlocks[j];
        avgMSE += MSE/n;
    }

    printf("\nsuccess percentage: %.2f\n",succ*100.0/numIter);
    avgMSE/=numIter;
    printf("\naverage MSE: %.15f\n",avgMSE);
    printf("\naverage k-LiMapS execution time: %.6f\n", avgt/numIter);

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
