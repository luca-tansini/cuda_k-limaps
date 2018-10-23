#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "vectorUtility.cu"

#ifndef _COMMON_H
    #include "common.h"
#endif

#define BLOCK_SIZE 256

//Funzione compare richiesta da qsort, ordine decrescente
//restituisce -1 se elem1 > elem2, 1 se elem2 > elem1, 0 se uguali
int comp(const void *elem1, const void *elem2) {
    double f = *((double*)elem1);
    double s = *((double*)elem2);
    if (f > s) return -1;
    if (f < s) return 1;
    return 0;
}

//Kernel che implementa lo shrinkage F(lambda): b = F(lambda,a)
__global__ void fShrinkage(double lambda, double *a, double *b, int len){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < len){
        b[tid] = a[tid] * (1 - powf(M_E, -lambda*fabsf(a[tid])));
    }
}

//Kernel che implementa il passo finale di thresholding
__global__ void thresholding(double *v, int len, double threshold){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < len)
        if(fabsf(v[tid]) < threshold)
            v[tid] = 0;
}

/*
Funzione che implementa l'algoritmo k-LiMapS.
Descrizione dei parametri:
    k: livello di sparsità
    D: il dizionario € R^(n*m)
    DINV: la pseudoinversa del dizionario (€ R^(m*n))
    s: il segnale € R^n
    alpha: il vettore di output € R^m
    maxIter: un limite al numero di iterazioni del ciclo interno
Tutti i puntatori passati si assume siano device memory e adeguatamente preallocati.
Tutte le matrici devono essere in formato column-major per compatibilità con le librerie cuBLAS
Il risultato è una soluzione approssimata per s = D*alpha t.c. lzero-norm(alpha) <= k
*/
void k_LiMapS(int k, double *D, int n, int m, double *DINV, double *s, double *alpha, int maxIter){

    //Crea il cublas handle
    cublasHandle_t handle;
	CHECK_CUBLAS(cublasCreate(&handle));

    //calcolca alpha inizile: alpha = D * s
    double cuAlpha = 1, cuBeta = 0;
    CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_N, m, n, &cuAlpha, DINV, m, s, 1, &cuBeta, alpha, 1));

    //Allocazioni per il ciclo interno
    int i = 0;
    int mBlocks = ceil(m*1.0/BLOCK_SIZE);
    double *sigma,*beta,*oldalpha,*tmp;
    dim3 dimBlock(BLOCK_SIZE,1,1);
    dim3 dimGridM(mBlocks,1,1);
    dim3 dimGridN(ceil(n*1.0/BLOCK_SIZE),1,1);

    CHECK(cudaMalloc(&beta, m*sizeof(double)));
    CHECK(cudaMalloc(&oldalpha, mBlocks*BLOCK_SIZE*sizeof(double)));
    CHECK(cudaMemset(oldalpha, 0, mBlocks*BLOCK_SIZE*sizeof(double)));
    CHECK(cudaMalloc(&tmp, m*sizeof(double)));
    CHECK(cudaMallocHost(&sigma, m*sizeof(double)));

    //Ciclo interno dell'algoritmo
    while(i < maxIter){

        //1a. recupera alpha dalla memoria device in sigma (host)
        CHECK_CUBLAS(cublasGetVector(m, sizeof(double), alpha, 1, sigma, 1));
        //1b. ordina i valori assoluti di sigma in ordine decrescente
        for(int j=0; j<m; j++)
            sigma[j] = fabs(sigma[j]);
        qsort(sigma, m, sizeof(double), comp);

        //2. calcola lambda = 1/sigma[k]
        double lambda = 1/sigma[k];

        //3. calcola beta = F(lambda, alpha)
        fShrinkage<<<dimGridM,dimBlock>>>(lambda, alpha, beta, m);
        CHECK(cudaDeviceSynchronize());

        //4. aggiorna alpha = beta - DINV * (D * beta - s)

        //salva oldalpha
        CHECK(cudaMemcpy(oldalpha, alpha, m*sizeof(double), cudaMemcpyDeviceToDevice));

        //alpha = D * beta (€ R^n)
        CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_N, n, m, &cuAlpha, D, n, beta, 1, &cuBeta, alpha, 1));

        //alpha = alpha - s (€ R^n)
        vectorSum<<<dimGridN,dimBlock>>>(1, alpha, -1, s, alpha, n);
        CHECK(cudaDeviceSynchronize());

        //tmp = DINV * alpha (€ R^m)
        CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_N, m, n, &cuAlpha, DINV, m, alpha, 1, &cuBeta, tmp, 1));

        //alpha = beta - tmp (€ R^m)
        vectorSum<<<dimGridM,dimBlock>>>(1, beta, -1, tmp, alpha, m);
        CHECK(cudaDeviceSynchronize());

        //aggiorna le condizioni del ciclo
        vectorSum<<<dimGridM,dimBlock>>>(1, alpha, -1, oldalpha, oldalpha, m);
        CHECK(cudaDeviceSynchronize());
        double norm = vectorNorm(oldalpha, m);
        if(norm <= 1e-5){
            break;
        }
        i++;
    }

    //step finale di thresholding: alpha[i] = 0 if |alpha[i]| <= sigma[k]
    thresholding<<<dimGridM,dimBlock>>>(alpha, m, sigma[k]);
    CHECK(cudaDeviceSynchronize());

    //Free della memoria
    CHECK(cudaFree(oldalpha));
    CHECK(cudaFree(beta));
    CHECK(cudaFree(tmp));
    CHECK(cudaFreeHost(sigma));
    CHECK_CUBLAS(cublasDestroy(handle));

}
