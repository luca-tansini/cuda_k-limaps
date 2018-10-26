#include "k-LiMapS.cu"
#include "vectorUtility.cu"
#include "MoorePenrose.cu"
#include "createDict.cu"

/*
Questo programma esegue i test "noiseless" per l'algoritmo k-LiMapS
Il dizionario D e il vettore k-sparso alphaopt vengono generati in GPU a partire da una distribuzione gaussiana.
La pseudoinversa viene calcolata in GPU con l'algoritmo di Jacobi per SVD.
Il segnale s viene calcolato come s = D * alphaopt.
Il programma, con numero di righe della matrice n preso in in input da riga di comando, esegue diversi test per valori di m = n,...,10n e k = 10%n,15%n,...,50%n
Per ogni tripla di valori n,m,k vengono eseguite 50 iterazioni, da cui vengono calcolati alcuni valori come:
    succ%:   una stima di quante volte l'algoritmo k-LiMapS ha prodotto una soluzione approssimata alphalimaps tale che la differenza di ogni elemento tra alphalimaps e alphaopt fosse al più 10^-3
    avgMSE:  la media sulle 50 iterazioni del MeanSquareError tra D*alphaopt e D*alphalimaps
    avgTime: la media sulle 50 iterazioni del tempo di calcolo dell'algoritmo k-LiMapS
Viene generato un nuovo dizionario per ogni valore di m, mentre alphaopt viene estratto ad ogni singola iterazione.
*/
int main(int argc, char **argv){

    if(argc != 2){
        printf("usage: noiselessTest <n>\n");
        return 2;
    }

    //Istruzione per flushare stdout ad ogni printf e non solo a fine riga
    setbuf(stdout, NULL);

    int n = atoi(argv[1]);

    printf("    n|     m| delta|     k|   rho|  succ%%|      avgMSE      | avgTime |\n");

    //Alloca i puntatori alla memoria device
    double *D,*DINV,*alphaopt,*s,*alphalimaps,*h_alphalimaps,*h_alphaopt;
    CHECK(cudaMalloc(&s, n*sizeof(double)));

    //Crea il cublas handle
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    double cualpha=1,cubeta=0;

    //Ciclo su m da n a 10n, step n
    for(int m = n; m <= 10*n; m += n){

        //Crea il dizionario
        CHECK(cudaMalloc(&D, n*m*sizeof(double)));
        createDict(D, n, m);

        //Calcola la pseudoinversa
        CHECK(cudaMalloc(&DINV, m*n*sizeof(double)));
        JacobiMoorePenroseInverse(D, n, m, DINV);
        if(!CheckPseudoinverse(D, n, m, DINV))
            printf("Something went wrong with the Moore-Penrose pseudoinverse!\n");

        //Alloca alphaopt e alphalimaps
        CHECK(cudaMalloc(&alphaopt, m*sizeof(double)));
        CHECK(cudaMalloc(&alphalimaps, m*sizeof(double)));
        CHECK(cudaMallocHost(&h_alphaopt, m*sizeof(double)));
        CHECK(cudaMallocHost(&h_alphalimaps, m*sizeof(double)));

        //Ciclo su k dal 10% di n al 50%, step 5%
        for(int l = 10; l<=50; l+=5){
            int k = n*l/100.0;

            int iters;
            int succ = 0;
            double avgMSE = 0;
            double avgTime = 0;

            //n, m, delta, k, rho
            printf("%5d| %5d| %5.2f| %5d| %5.2f| ", n, m, n*1.0/m, k, k*1.0/n);

            for(iters=0; iters<50; iters++){

                //Genera alphaopt
                generateAlpha(alphaopt, m, k);

                //Calcola s = D * alphaopt
                CHECK_CUBLAS(cublasDgemv(cublasHandle, CUBLAS_OP_N, n, m, &cualpha, D, n, alphaopt, 1, &cubeta, s, 1));

                //Chiama K_LiMapS
                double t=seconds();
                k_LiMapS(k, D, n, m, DINV, s, alphalimaps, 1000);
                avgTime += seconds() - t;

                //Check del risultato (succ%)
                CHECK(cudaMemcpy(h_alphaopt, alphaopt, m*sizeof(double), cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpy(h_alphalimaps, alphalimaps, m*sizeof(double), cudaMemcpyDeviceToHost));

                int i;
                for(i=0; i<m; i++)
                    if(fabs(h_alphaopt[i] - h_alphalimaps[i]) > 1e-3)
                        break;
                if(i == m)
                    succ++;

                //Calcola MSE
                avgMSE += MSE(s,D,alphalimaps,n,m);

            }

            avgMSE  /= iters;
            avgTime /= iters;

            //succ%, avgMSE, avgTime
            printf("%6.2f| %17.15f| %8.6f|\n", succ*100.0/50, avgMSE, avgTime);

        }

        CHECK(cudaFree(D));
        CHECK(cudaFree(DINV));
        CHECK(cudaFree(alphaopt));
        CHECK(cudaFree(alphalimaps));
        CHECK(cudaFreeHost(h_alphaopt));
        CHECK(cudaFreeHost(h_alphalimaps));

    }

    CHECK(cudaFree(s));

    return 0;

}
