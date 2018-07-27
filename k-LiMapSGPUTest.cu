#include "k-LiMapS.cu"
#include "MoorePenrose.cu"

/*
This main tests the k-LiMapS algorithm generating the dictionary and its pseudoinverse in GPU.
At the moment MoorePenrose.cu has proven wrong (due to the limitation of the cuSOLVER libraries)
*/
int main(int argc, char **argv){

    if(argc != 3){
        printf("usage: simpleTest n k\n");
        exit(2);
    }

    int n,k,m,i,j;
    n = atoi(argv[1]);
    k = atoi(argv[2]);
    m = n*k;

    srand(time(NULL));

    //Fill theta with random values between 0 and 1
    float *theta;
    CHECK(cudaMallocHost(&theta, n*m*sizeof(float)));
    for(i=0; i<n*m; i++)
        theta[i] = rand()/(float)RAND_MAX;

    //Fill optimal solution alpha with k random values
    float *alpha;
    CHECK(cudaMallocHost(&alpha, m*sizeof(float)));
    memset(alpha,0,m*sizeof(float));
    for(i=0; i<k; i++){
        j = rand()%m;
        if(alpha[j] != 0)
            i--;
        else
            alpha[j] = rand()/(float)RAND_MAX;
    }

    //calculate theta Moore-Penrose inverse
    float *d_theta,*d_thetaPseudoInv;
    CHECK(cudaMalloc(&d_theta, n*m*sizeof(float)));
    CHECK(cudaMemcpy(d_theta, theta, n*m*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_thetaPseudoInv, m*n*sizeof(float)));
    MoorePenroseInverse(d_theta, n, m, d_thetaPseudoInv);

    float *thetaPseudoInv;
    CHECK(cudaMallocHost(&thetaPseudoInv, m*n*sizeof(float)));
    CHECK(cudaMemcpy(thetaPseudoInv, d_thetaPseudoInv, m*n*sizeof(float), cudaMemcpyDeviceToHost));

    //Calculate b = theta * alpha
    float *d_b,*d_alpha,cualpha=1,cubeta=0;
    cublasHandle_t cublasHandle;

    CHECK(cudaMalloc(&d_b, n*sizeof(float)));
    CHECK(cudaMalloc(&d_alpha, m*sizeof(float)));
    CHECK(cudaMemcpy(d_alpha, alpha, m*sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    CHECK_CUBLAS(cublasSgemv(cublasHandle, CUBLAS_OP_N, n, m, &cualpha, d_theta, n, d_alpha, 1, &cubeta, d_b, 1));

    //call k_LiMapS

    //For compatibility with future use, k_LiMapS parameters are supposed to be host memory pointers, so we need to transfer MoorePenroseInverse result and d_b into host memory
    float *b, *limapsAlpha;
    CHECK(cudaMallocHost(&b, n*sizeof(float)));
    CHECK(cudaMemcpy(b, d_b, n*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMallocHost(&limapsAlpha, m*sizeof(float)));

    k_LiMapS(k, theta, n, m, thetaPseudoInv, d_b, limapsAlpha, 10);

    //Check result
    for(i=0; i<m; i++)
        if(abs(alpha[i] - limapsAlpha[i]) > 1e-4)
            break;

    if(i < m){
        printf("NOPE\n");
        printf("\nalpha:\n");
        printHighlightedVector(alpha, m);

        printf("\nlimapsAlpha:\n");
        printHighlightedVector(limapsAlpha, m);
    }
    else
        printf("OK!\n");

    //Free memory
    CHECK(cudaFreeHost(theta));
    CHECK(cudaFreeHost(alpha));
    CHECK(cudaFreeHost(limapsAlpha));
    CHECK(cudaFreeHost(thetaPseudoInv));
    CHECK(cudaFree(d_theta));
    CHECK(cudaFree(d_alpha));
    CHECK(cudaFree(d_thetaPseudoInv));
    CHECK(cudaFree(d_b));
    cudaDeviceReset();

    return 0;

}
