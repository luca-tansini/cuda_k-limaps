#include "k-LiMapS.cu"
#include "vectorUtility.cu"
#include "MoorePenrose.cu"
#include "createDict.cu"

int main(int argc, char **argv){

    if(argc != 3){
        printf("usage: noiselessTest <n> <k>\n");
        return 2;
    }

    setbuf(stdout, NULL);

    int n = atoi(argv[1]);

    //ALLOCATE MEMORY POINTERS
    double *D,*DINV,*alphaopt,*s,*alphalimaps,*h_alphalimaps,*h_alphaopt;
    CHECK(cudaMalloc(&s, n*sizeof(double)));

    //CREATE CUBLAS HANDLE
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    double cualpha=1,cubeta=0;

    //CICLO SU M
    for(int m = n; m <= 2*n; m += n/10){

        //CREA DIZIONARIO
        CHECK(cudaMalloc(&D, n*m*sizeof(double)));
        createDict(D, n, m);

        //CALCOLA PSEUDOINVERSA
        CHECK(cudaMalloc(&DINV, m*n*sizeof(double)));
        JacobiMoorePenroseInverse(D, n, m, DINV);
        if(!CheckPseudoinverse(D, n, m, DINV))
            printf("Something went wrong with the Moore-Penrose pseudoinverse!\n");

        //ALLOCA alphaopt e alphalimaps
        CHECK(cudaMalloc(&alphaopt, m*sizeof(double)));
        CHECK(cudaMalloc(&alphalimaps, m*sizeof(double)));
        CHECK(cudaMallocHost(&h_alphaopt, m*sizeof(double)));
        CHECK(cudaMallocHost(&h_alphalimaps, m*sizeof(double)));

        //CICLO SU K e iters
        for(int k=n/10; k<=n/2; k+=n/20){
            for(int iters=0; iters<50; iters++){

                //GENERA alphaopt
                generateAlpha(alphaopt, m, k);

                //CALCOLA s = D * alphaopt
                CHECK_CUBLAS(cublasDgemv(cublasHandle, CUBLAS_OP_N, n, m, &cualpha, D, n, alphaopt, 1, &cubeta, s, 1));

                //CHIAMA K_LiMapS
                devMemK_LiMapS(k, D, n, m, DINV, s, alphalimaps, 1000);

                //CHECK DEL RISULTATO
                CHECK(cudaMemcpy(h_alphaopt, alphaopt, m*sizeof(double), cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpy(h_alphalimaps, alphalimaps, m*sizeof(double), cudaMemcpyDeviceToHost));

                int i;
                for(i=0; i<m; i++)
                    if(fabs(h_alphaopt[i] - h_alphalimaps[i]) > 1e-3)
                        break;
                if(i == m)
                    printf("ALL GOOD\n");
                else
                    printf("SOMETHING WENT WRONG!\n");

                printf("MSE: %f\n", MSE(s,D,alphalimaps,n,m));

            }
        }

        CHECK(cudaFree(D));
        CHECK(cudaFree(DINV));
        CHECK(cudaFree(alphaopt));
        CHECK(cudaFree(alphalimaps));
        CHECK(cudaFreeHost(h_alphaopt));
        CHECK(cudaFreeHost(h_alphalimaps));

    }

    //FREE
    CHECK(cudaFree(s));

    return 0;

}
