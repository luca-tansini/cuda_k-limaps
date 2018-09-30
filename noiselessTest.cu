#include "k-LiMapS.cu"
#include "vectorUtility.cu"
#include "MoorePenrose.cu"
#include "createDict.cu"

/*
Function calculating MSE: sum((s - D * alphalimaps)^2)/n
*/
double MSE(double *s, double *D, double *alpa, int n, int m){

    int blocks = ceil(n*1.0/BLOCK_SIZE);
    dim3 dimGrid(blocks,1,1);
    dim3 dimBlock(BLOCK_SIZE,1,1);
    double *limapsS,*partialMSEBlocks;

    CHECK(cudaMalloc(&limapsS, blocks*BLOCK_SIZE*sizeof(double)));
    CHECK(cudaMemset(limapsS, 0, blocks*BLOCK_SIZE*sizeof(double)));
    CHECK(cudaMallocHost(&partialMSEBlocks, blocks*sizeof(double)));

    //limapsS = D * alphalimaps
    CHECK_CUBLAS(cublasDgemv(cublasHandle, CUBLAS_OP_N, n, m, &cualpha, D, n, alphalimaps, 1, &cubeta, limapsS, 1));

    //limapsS = s - limapsS
    vectorSum<<<dimGrid,dimBlock>>>(1, s, -1, limapsS, limapsS, n);
    CHECK(cudaDeviceSynchronize());

    vector2norm<<<dimGrid,dimBlock>>>(d_limapsB);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(partialMSEBlocks, d_limapsB, blocks * sizeof(double), cudaMemcpyDeviceToHost));
    double MSE = 0;
    for(j=0; j<blocks; j++)
        MSE += partialMSEBlocks[j];
    MSE /= n;

    CHECK(cudaFree(limapsS));
    CHECK(cudaFreeHost(partialMSEBlocks));

    return MSE;
}

int main(int argc, char **argv){

    if(argc != 2){
        printf("usage: noiselessTest <n>\n");
        return 2;
    }

    setbuf(stdout, NULL);

    int n = atoi(argv[1]);
    int m = 2*n;
    int k = n/4;

    //DEBUG
    printf("n:%d\tm:%d\tk:%d\n",n,m,k);
    //END DEBUG

    //CREA DIZIONARIO D
    printf("creating dictionary D...");
    double *D;
    CHECK(cudaMalloc(&D, n*m*sizeof(double)));

    createDict(D, n, m);
    printf("done\n");

    //CALCOLA PSEUDOINVERSA DINV
    printf("computing pseudoinverse DINV...");
    double *DINV;
    CHECK(cudaMalloc(&DINV, m*n*sizeof(double)));

    JacobiMoorePenroseInverse(D, n, m, DINV);
    printf("done\n");

    if(!CheckPseudoinverse(D, n, m, DINV))
        printf("Something went wrong with the Moore-Penrose pseudoinverse!\n");

    //GENERA ALPHAOPT
    printf("generating alphaopt...");
    double *alphaopt;
    CHECK(cudaMalloc(&alphaopt, m*sizeof(double)));

    generateAlpha(alphaopt, m, k);
    printf("done\n");

    //CALCOLA s = D * alphaopt
    printf("computing s = D * alphaopt...");
    double *s;
    CHECK(cudaMalloc(&s, n*sizeof(double)));

    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    double cualpha=1,cubeta=0;

    CHECK_CUBLAS(cublasDgemv(cublasHandle, CUBLAS_OP_N, n, m, &cualpha, D, n, alphaopt, 1, &cubeta, s, 1));
    printf("done\n");

    //CHIAMA K_LiMapS
    printf("calling K_LiMapS...");
    double *alphalimaps;
    CHECK(cudaMalloc(&alphalimaps, m*sizeof(double)));

    devMemK_LiMapS(k, D, n, m, DINV, s, alphalimaps, 1000);
    printf("done\n");

    //CHECK DEL RISULTATO
    double *h_alphalimaps,*h_alphaopt;
    CHECK(cudaMallocHost(&h_alphaopt, m*sizeof(double)));
    CHECK(cudaMallocHost(&h_alphalimaps, m*sizeof(double)));
    CHECK(cudaMemcpy(h_alphaopt, alphaopt, m*sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_alphalimaps, alphalimaps, m*sizeof(double), cudaMemcpyDeviceToHost));

    int i;
    for(i=0; i<m; i++)
        if(fabs(h_alphaopt[i] - h_alphalimaps[i]) > 1e-3)
            break;
    if(i == m)
        printf("ALL GOOD\n");
    else{
        printf("SOMETHING WENT WRONG!\n");

        printf("alphaopt:\n");
        printHighlightedVector(h_alphaopt, m);
        printf("\n");

        printf("alphalimaps:\n");
        printHighlightedVector(h_alphalimaps, m);
        printf("\n");
    }

    printf("MSE: %f\n", MSE(s,D,alphalimaps,n,m));

    //FREE
    CHECK(cudaFree(D));
    CHECK(cudaFree(DINV));
    CHECK(cudaFree(alphaopt));
    CHECK(cudaFree(s));
    CHECK(cudaFree(alphalimaps));

    return 0;

}
