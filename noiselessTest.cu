#include "k-LiMapS.cu"
#include "vectorUtility.cu"
#include "MoorePenrose.cu"
#include "createDict.cu"

int main(int argc, char **argv){

    if(argc != 2){
        printf("usage: noiselessTest <n>\n");
        return 2;
    }

    setbuf(stdout, NULL);

    int n = atoi(argv[1]);

    printf("    n|     m| delta|     k|   rho|  succ%%|      avgMSE      | avgTime |\n");

    //ALLOCATE MEMORY POINTERS
    double *D,*DINV,*alphaopt,*s,*alphalimaps,*h_alphalimaps,*h_alphaopt;
    CHECK(cudaMalloc(&s, n*sizeof(double)));

    //CREATE CUBLAS HANDLE
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    double cualpha=1,cubeta=0;

    //CICLO SU M DA N A 5N, STEP N
    for(int m = n; m <= 5*n; m += n){

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

        //CICLO SU K DAL 10% DI N AL 50%, STEP 5%
        for(int l = 10; l<=50; l+=5){
            int k = n*l/100.0;

            int iters;
            int succ = 0;
            double avgMSE = 0;
            double avgTime = 0;

            //n, m, delta, k, rho
            printf("%5d| %5d| %5.2f| %5d| %5.2f| ", n, m, n*1.0/m, k, k*1.0/n);

            for(iters=0; iters<50; iters++){

                //GENERA alphaopt
                generateAlpha(alphaopt, m, k);

                //CALCOLA s = D * alphaopt
                CHECK_CUBLAS(cublasDgemv(cublasHandle, CUBLAS_OP_N, n, m, &cualpha, D, n, alphaopt, 1, &cubeta, s, 1));

                //CHIAMA K_LiMapS
                double t=seconds();
                devMemK_LiMapS(k, D, n, m, DINV, s, alphalimaps, 1000);
                avgTime += seconds() - t;

                //CHECK DEL RISULTATO
                CHECK(cudaMemcpy(h_alphaopt, alphaopt, m*sizeof(double), cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpy(h_alphalimaps, alphalimaps, m*sizeof(double), cudaMemcpyDeviceToHost));

                int i;
                for(i=0; i<m; i++)
                    if(fabs(h_alphaopt[i] - h_alphalimaps[i]) > 1e-3)
                        break;
                if(i == m)
                    succ++;

                avgMSE += MSE(s,D,alphalimaps,n,m);

            }

            avgMSE  /= iters;
            avgTime /= iters;

            //succ, avgMSE, avgTime
            printf("%6.2f| %17.15f| %8.6f|\n", succ*100.0/50, avgMSE, avgTime);

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
