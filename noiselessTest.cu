#include "k-LiMapS.cu"
#include "MoorePenrose.cu"

//Function generating the dictionary
void createDict(double *D, int n, int m){

    int i;

    srand(time(NULL));

    for(i=0;i<n*m;i++)
        D[i] = rand()/(double)RAND_MAX;

}

int main(int argc, char **argv){

    if(argc != 2){
        printf("usage: noiselessTest <n>\n");
        return 2;
    }

    int n = atoi(argv[1]);
    int m = 2*n;
    int k = n/4;

    //DEBUG
    printf("n:%d\tm:%d\tk:%d\n",n,m,k);
    //END DEBUG

    //CREA DIZIONARIO D
    double *h_D,*d_D;
    CHECK(cudaMallocHost(&h_D, n*m*sizeof(double)));

    createDict(h_D, n, m);

    CHECK(cudaMalloc(&d_D, n*m*sizeof(double)));
    CHECK(cudaMemcpy(d_D, h_D, n*m*sizeof(double), cudaMemcpyHostToDevice));

    /*//DEBUG
    printf("D:\n");
    printColumnMajorMatrixForPython(h_D, n, m);
    printf("\n");
    //END DEBUG*/

    //CALCOLA PSEUDOINVERSA DINV
    double *h_DINV,*d_DINV;
    CHECK(cudaMalloc(&d_DINV, m*n*sizeof(double)));

    TransposedMoorePenroseInverse(d_D, n, m, d_DINV);

    if(!CheckPseudoInverse(d_D, n, m, d_DINV)){
        printf("Something went wrong with the Moore-Penrose pseudoinverse!\n");
        //return 3;
    }

    CHECK(cudaMallocHost(&h_DINV, m*n*sizeof(double)));
    CHECK(cudaMemcpy(h_DINV, d_DINV, m*n*sizeof(double), cudaMemcpyDeviceToHost));

    /*//DEBUG
    printf("DINV:\n");
    printColumnMajorMatrixForPython(h_DINV, m, n);
    printf("\n");
    //END DEBUG*/

    //GENERA ALPHAOPT
    double *h_alphaopt,*d_alphaopt;
    CHECK(cudaMallocHost(&h_alphaopt, m*sizeof(double)));

    int i,j;
    memset(h_alphaopt, 0, m*sizeof(double));
    for(i=0; i<k; i++){
        j = rand()%m;
        if(h_alphaopt[j] != 0)
            i--;
        else
            h_alphaopt[j] = rand()/(double)RAND_MAX;
    }

    CHECK(cudaMalloc(&d_alphaopt, m*sizeof(double)));
    CHECK(cudaMemcpy(d_alphaopt, h_alphaopt, m*sizeof(double), cudaMemcpyHostToDevice));

    /*//DEBUG
    printf("alphaopt:\n");
    printColumnMajorMatrixForPython(h_alphaopt, m, 1);
    printf("\n");
    //END DEBUG*/

    //GENERA S = D * alphaopt
    double *h_s,*d_s;
    CHECK(cudaMalloc(&d_s, n*sizeof(double)));

    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    double cualpha=1,cubeta=0;

    CHECK_CUBLAS(cublasDgemv(cublasHandle, CUBLAS_OP_N, n, m, &cualpha, d_D, n, d_alphaopt, 1, &cubeta, d_s, 1));

    CHECK(cudaMallocHost(&h_s, n*sizeof(double)));
    CHECK(cudaMemcpy(h_s, d_s, n*sizeof(double), cudaMemcpyDeviceToHost));

    /*//DEBUG
    printf("s:\n");
    printColumnMajorMatrixForPython(h_s, n, 1);
    printf("\n");
    //END DEBUG*/

    //CHIAMA K_LiMapS
    double *h_alphalimaps,*d_alphalimaps;
    CHECK(cudaMalloc(&d_alphalimaps, m*sizeof(double)));

    devMemK_LiMapS(k, d_D, n, m, d_DINV, d_s, d_alphalimaps, 1000);

    CHECK(cudaMallocHost(&h_alphalimaps, m*sizeof(double)));
    CHECK(cudaMemcpy(h_alphalimaps, d_alphalimaps, m*sizeof(double), cudaMemcpyDeviceToHost));

    /*//DEBUG
    printf("alphalimaps:\n");
    printColumnMajorMatrixForPython(h_alphalimaps, m, 1);
    printf("\n");
    //END DEBUG*/

    //CHECK DEL RISULTATO
    for(i=0; i<m; i++)
        if(fabs(h_alphaopt[i] - h_alphalimaps[i]) > 1e-3)
            break;
    if(i == m)
        printf("ALL GOOD\n");
    else
        printf("SOMETHING WENT WRONG!\n");

    //FREE
    CHECK(cudaFreeHost(h_D));
    CHECK(cudaFree(d_D));
    CHECK(cudaFreeHost(h_DINV));
    CHECK(cudaFree(d_DINV));
    CHECK(cudaFreeHost(h_alphaopt));
    CHECK(cudaFree(d_alphaopt));
    CHECK(cudaFreeHost(h_s));
    CHECK(cudaFree(d_s));
    CHECK(cudaFreeHost(h_alphalimaps));
    CHECK(cudaFree(d_alphalimaps));

    return 0;

}
