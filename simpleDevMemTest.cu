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

/*
This main tests the k-LiMapS algorithm generating the dictionary and its pseudoinverse in GPU.
*/
int main(int argc, char **argv){

    if(argc != 5){
        printf("usage: simpleDevMemTest n m k numIter\n");
        exit(-1);
    }

    int n,m,k,i,j,numIter;
    n = atoi(argv[1]);
    m = atoi(argv[2]);
    k = atoi(argv[3]);
    numIter = atoi(argv[4]);

    double t1,avgt=0;

    srand(time(NULL));

    t1 = seconds();

    float *theta, *alpha, *limapsAlpha, *d_limapsAlpha;
    CHECK(cudaMallocHost(&theta, n*m*sizeof(float)));
    CHECK(cudaMallocHost(&alpha, m*sizeof(float)));
    CHECK(cudaMallocHost(&limapsAlpha, m*sizeof(float)));
    CHECK(cudaMalloc(&d_limapsAlpha, m*sizeof(float)));

    //Fill theta with random values between 0 and 1
    for(i=0; i<n*m; i++)
        theta[i] = rand()/(float)RAND_MAX;

    //calculate theta Moore-Penrose inverse
    float *d_theta,*d_thetaPseudoInv;
    CHECK(cudaMalloc(&d_theta, n*m*sizeof(float)));
    CHECK(cudaMemcpy(d_theta, theta, n*m*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_thetaPseudoInv, m*n*sizeof(float)));
    TransposedMoorePenroseInverse(d_theta, n, m, d_thetaPseudoInv);

    printf("Dictionary generation and pseudoinverse computation elapsed time: %.6f\n", seconds()-t1);

    //Initialize cublas
    float cualpha=1,cubeta=0;
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    //Allocate device pointers
    float *d_b, *d_alpha;
    CHECK(cudaMalloc(&d_b, n*sizeof(float)));
    CHECK(cudaMalloc(&d_alpha, m*sizeof(float)));

    int succ = 0;
    for(int iter = 0; iter < numIter; iter++ ){

        //Fill optimal solution alpha with k random values
        memset(alpha,0,m*sizeof(float));
        for(i=0; i<k; i++){
            j = rand()%m;
            if(alpha[j] != 0)
                i--;
            else
                alpha[j] = rand()/(float)RAND_MAX;
        }

        CHECK(cudaMemcpy(d_alpha, alpha, m*sizeof(float), cudaMemcpyHostToDevice));

        //Calculate b = theta * alpha
        CHECK_CUBLAS(cublasSgemv(cublasHandle, CUBLAS_OP_N, n, m, &cualpha, d_theta, n, d_alpha, 1, &cubeta, d_b, 1));

        t1 = seconds();
        //call k_LiMapS
        k_LiMapS(k, d_theta, n, m, d_thetaPseudoInv, d_b, d_limapsAlpha, 1000);

        //Retrieve result
        CHECK(cudaMemcpy(limapsAlpha, d_limapsAlpha, m*sizeof(float), cudaMemcpyHostToDevice));

        avgt += seconds() - t1;

        //Check result
        for(i=0; i<m; i++)
            if(abs(alpha[i] - limapsAlpha[i]) > 1e-4)
                break;
        if(i == m)
            succ++;

    }

    printf("\n%.2f%%\n",100.0*succ/numIter);

    printf("\naverage k-LiMapS execution time: %.6f\n", avgt/numIter);

    //Free memory
    CHECK(cudaFreeHost(theta));
    CHECK(cudaFreeHost(alpha));
    CHECK(cudaFreeHost(limapsAlpha));
    CHECK(cudaFree(d_theta));
    CHECK(cudaFree(d_alpha));
    CHECK(cudaFree(d_thetaPseudoInv));
    CHECK(cudaFree(d_b));
    cudaDeviceReset();

    return 0;

}
