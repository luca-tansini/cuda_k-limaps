#include <float.h>
#include <cusolverDn.h>

#ifndef _COMMON_H
    #include "common.h"
#endif

#ifndef _MATRIX_PRINT_H
    #include "matrixPrint.h"
#endif

#define BLOCK_SIZE 256

/*
Kernel che prende il vettore di valori singolari S (di lunghezza m) e produce la pseudo inversa della matrice diagonale di S. Poichè [n x m] sarebbe la dimensione della matrice diagonale di S, e nel calcolo della pseudoinversa dobbiamo anche trasporre la matrice, la pseudoinversa avrà dimensione [m x n], con leading dimension m.
Gli elementi della diagonale devono essere invertiti solo se diversi da 0. Per determinare che cosa è zero, usiamo la costante DBL_EPSILON della macchina.
*/
__global__ void calculateDiagPinv(double *S, double *Spinv, int n, int m){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < m && idx < n){
        if(fabsf(S[idx]) > DBL_EPSILON)
            Spinv[idx*m + idx] = 1/S[idx];
        else
            Spinv[idx*m + idx] = 0;
    }
}

/*
Questa funzione calcola la pseudoinversa di Moore-Penrose della matrice in input A (n*m, con n > m), restituendo il risultato in Apinv (che si assume sia preallocato).

La pseudoinversa è calcolata tramite decomposizione SVD.
Se SVD(A) = U*S*V^T --> A^+ = V * S^+ * U^T, dove S^+ è ottenuta sostituendo ogni elemento non-zero della diagonale con il suo reciproco e trasponendo.

Le librerie cuSOLVER usate per calcolare l'SVD hanno bisogno che la matrice in input sia n x m con n >= m.
*/
void MoorePenroseInverse(double *A, int n, int m, double *Apinv){

    if(n < m){
        printf("error: n must be >= m! (you can transpose the input matrix and then transpose the result to work with matrices that have less rows than columns)\n");
        return;
    }

    cusolverDnHandle_t cusolverHandle;
    cublasHandle_t cublasHandle;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverHandle));
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    //Calcola la dimensione per il buffer  di lavoro e lo alloca
    int bufferDim;
    double *buffer;
    CHECK_CUSOLVER(cusolverDnDgesvd_bufferSize(cusolverHandle, n, m, &bufferDim));
    CHECK(cudaMalloc(&buffer,bufferDim*sizeof(double)));

    //Alloca U,S,VT
    double *U,*S,*VT;
    CHECK(cudaMalloc(&U, n*n*sizeof(double)));
    CHECK(cudaMalloc(&S, m*sizeof(double)));
    CHECK(cudaMalloc(&VT, m*m*sizeof(double)));

    //Calcola SVD con cuSOLVER
    double *Acopy; //usiamo una copia di A perchè gesvd distrugge la matrice in input
    CHECK(cudaMalloc(&Acopy, n*m*sizeof(double)));
    CHECK(cudaMemcpy(Acopy, A, n*m*sizeof(double), cudaMemcpyDeviceToDevice));

    int *dev_info, h_dev_info;
    CHECK(cudaMalloc(&dev_info, sizeof(int)));
    CHECK_CUSOLVER(cusolverDnDgesvd(cusolverHandle, 'A', 'A', n, m, Acopy, n, S, U, n, VT, m, buffer, bufferDim, NULL, dev_info));
    CHECK(cudaMemcpy(&h_dev_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost));
    if(h_dev_info != 0)
        printf("Something went wrong (dev_info=%d)\n", h_dev_info);

    //Calcola S^+
    double *Spinv;
    CHECK(cudaMalloc(&Spinv, m*n*sizeof(double)));
    CHECK(cudaMemset(Spinv, 0, m*n*sizeof(double)));

    dim3 dimBlock(BLOCK_SIZE,1,1);
    dim3 dimGrid(ceil(m*1.0/BLOCK_SIZE),1,1);
    calculateDiagPinv<<<dimGrid,dimBlock>>>(S, Spinv, n, m);
    CHECK(cudaDeviceSynchronize());

    //Calcola Apinv = VT^T * S^+ * U^T
    double alpha=1,beta=0,*tmp;
    CHECK(cudaMalloc(&tmp, m*n*sizeof(double)));

    //tmp = VT^T * S^+
    CHECK_CUBLAS(cublasDgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, m, &alpha, VT, m, Spinv, m, &beta, tmp, m));

    //Apinv = tmp * U^T
    CHECK_CUBLAS(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, n, &alpha, tmp, m, U, n, &beta, Apinv, m));

    //Free della memoria
    CHECK(cudaFree(buffer));
    CHECK(cudaFree(U));
    CHECK(cudaFree(S));
    CHECK(cudaFree(Spinv));
    CHECK(cudaFree(VT));
    CHECK(cudaFree(tmp));
    CHECK_CUBLAS(cublasDestroy(cublasHandle));
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverHandle));

}

/*
Questa versione usa cusolverDnSgesvdj che usa il metodo di Jacobi per la SVD decomposition.
Non ci sono vincoli su n e m.
cusolverDnSgesvdj restituisce V invece che VH
*/
void JacobiMoorePenroseInverse(double *A, int n, int m, double *Apinv){

    cusolverDnHandle_t cusolverHandle;
    cublasHandle_t cublasHandle;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverHandle));
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    //Alloca U,S,V
    double *U,*S,*V;
    CHECK(cudaMalloc(&U, n*n*sizeof(double)));
    CHECK(cudaMalloc(&S, m*sizeof(double)));
    CHECK(cudaMalloc(&V, m*m*sizeof(double)));

    //Calcola SVD con cuSOLVER
    double *Acopy; //usiamo una copia di A perchè gesvdj distrugge la matrice in input
    CHECK(cudaMalloc(&Acopy, n*m*sizeof(double)));
    CHECK(cudaMemcpy(Acopy, A, n*m*sizeof(double), cudaMemcpyDeviceToDevice));

    //Set up dei parametri di cusolverDnDgesvdj
    int bufferDim;
    double *buffer;
    gesvdjInfo_t gesvdj_params = NULL; //parametri di default
    cusolverDnCreateGesvdjInfo(&gesvdj_params);
    CHECK_CUSOLVER(cusolverDnDgesvdj_bufferSize(cusolverHandle, CUSOLVER_EIG_MODE_VECTOR, 0, n, m, Acopy, n, S, U, n, V, m, &bufferDim, gesvdj_params));
    CHECK(cudaMalloc(&buffer,bufferDim*sizeof(double)));

    //Chiama cusolverDnDgesvdj
    int *dev_info, h_dev_info;
    CHECK(cudaMalloc(&dev_info, sizeof(int)));
    CHECK_CUSOLVER(cusolverDnDgesvdj(cusolverHandle, CUSOLVER_EIG_MODE_VECTOR, 0, n, m, Acopy, n, S, U, n, V, m, buffer, bufferDim, dev_info, gesvdj_params));
    CHECK(cudaMemcpy(&h_dev_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost));
    if(h_dev_info != 0)
        printf("Something went wrong (dev_info=%d)\n", h_dev_info);

    //Calcola S^+
    double *Spinv;
    CHECK(cudaMalloc(&Spinv, m*n*sizeof(double)));
    CHECK(cudaMemset(Spinv, 0, m*n*sizeof(double)));

    dim3 dimBlock(BLOCK_SIZE,1,1);
    dim3 dimGrid(ceil(m*1.0/BLOCK_SIZE),1,1);
    calculateDiagPinv<<<dimGrid,dimBlock>>>(S, Spinv, n, m);
    CHECK(cudaDeviceSynchronize());

    //Calcola Apinv = V * S^+ * U^T
    double alpha=1,beta=0,*tmp;
    CHECK(cudaMalloc(&tmp, m*n*sizeof(double)));

    //tmp = V * S^+
    CHECK_CUBLAS(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, m, &alpha, V, m, Spinv, m, &beta, tmp, m));

    //Apinv = tmp * U^T
    CHECK_CUBLAS(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, n, &alpha, tmp, m, U, n, &beta, Apinv, m));

    //Free della memoria
    CHECK(cudaFree(buffer));
    CHECK(cudaFree(U));
    CHECK(cudaFree(S));
    CHECK(cudaFree(Spinv));
    CHECK(cudaFree(V));
    CHECK(cudaFree(tmp));
    CHECK_CUBLAS(cublasDestroy(cublasHandle));
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverHandle));

}

/*
Funzione che verifica la correttezza del calcolo della pseudoinversa.
Il controllo effettuato si basa sulle proprietà della pseudoinversa ed è: A * Apinv * A =?= A
*/
int CheckPseudoinverse(double *A, int n, int m, double *Apinv){

    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    //tmp = A * Apinv
    double *tmp,alpha=1,beta=0;
    CHECK(cudaMalloc(&tmp, n*n*sizeof(double)));

    CHECK_CUBLAS(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, m, &alpha, A, n, Apinv, m, &beta, tmp, n));

    //tmp2 = tmp * A
    double *tmp2;
    CHECK(cudaMalloc(&tmp2, n*m*sizeof(double)));

    CHECK_CUBLAS(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, n, &alpha, tmp, n, A, n, &beta, tmp2, n));

    //tmp2 =?= A
    double *h_tmp2,*h_A;
    CHECK(cudaMallocHost(&h_tmp2, n*m*sizeof(double)));
    CHECK(cudaMemcpy(h_tmp2, tmp2, n*m*sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMallocHost(&h_A, n*m*sizeof(double)));
    CHECK(cudaMemcpy(h_A, A, n*m*sizeof(double), cudaMemcpyDeviceToHost));

    int i;
    for(i=0; i<n*m; i++)
        if(fabs(h_A[i] - h_tmp2[i]) > 1e-5){
            printf("at index %d diff is: %f\n",i, h_A[i] - h_tmp2[i]);
            break;
        }

    int ret=1;
    if(i < n*m)
        ret = 0;

    CHECK(cudaFree(tmp));
    CHECK(cudaFree(tmp2));
    CHECK(cudaFreeHost(h_tmp2));
    CHECK(cudaFreeHost(h_A));
    CHECK_CUBLAS(cublasDestroy(cublasHandle));

    return ret;
}
