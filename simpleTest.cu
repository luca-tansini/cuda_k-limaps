#include "k-LiMapS.cu"
#include <cusolverDn.h>

/*
This function calcuates the Moore-Penrose inverse matrix of the input matrix A (n*m, with m > n), leaving the result in apseudoinv, assumed preallocated.
The pseudoinverse is computed via SVD.
If SVD(A) = U*S*V^T --> A^+ = V * S^+ * U^T, where S^+ is obtained replacing each non-zero element on the diagonal with its reciprocal.
The cuSOLVER libraries used to calculate the SVD need the input matrix to be n x m with n < m, so we need to transpose our matrix
*/
void MoorePenroseInverse(float *A, int n, int m, float *Apseudoinv){

    //Calculate theta SVD via cuSOLVER api
    cusolverDnHandle_t cusolverHandle;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverHandle));
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    //transpose matrix A
    float alpha=1,beta=0;
    CHECK_CUBLAS(cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &alpha, A, m, &beta, A, n, Apseudoinv, n)); //non so se mettere n o m come ultimo parametro

    //Get dimension needed for the workspace buffer and allocate it
    int bufferDim;
    CHECK_CUSOLVER(cusolverDnSgesvd_bufferSize(cusolverHandle, m, n, &bufferDim));
    float *buffer;
    CHECK(cudaMalloc(&buffer,bufferDim));

    //Remembering A was transposed we have that SVD --> A[m*n] = U[m*m] * S[m*n diag] * V_T[n*n]
    //Allocate U,S,V_T
    float *U,*S,*V_T;
    CHECK(cudaMalloc(&U, m*m*sizeof(float)));
    //S that should be a diagonal matrix is returned as a simple vector instead
    CHECK(cudaMalloc(&S, m*sizeof(float)));
    CHECK(cudaMalloc(&V_T, n*n*sizeof(float)));

    //Calculate SVD (of A^T, we will have to do some considerations on our results)
    int success=0;
    CHECK_CUSOLVER(cusolverDnSgesvd(cusolverHandle, 'A', 'A', m, n, Apseudoinv, m, S, U, m, V_T, n, buffer, bufferDim, NULL, &success));

    if(success == 0)
        printf("Success!\n");
    else
        printf("Something went wrong (dev_info=%d)\n",success);

    //DEBUG
    //retrieve results
    float *h_U, *h_S, *h_V_T;

    CHECK(cudaMallocHost(&h_U,m*m*sizeof(float)));
    CHECK(cudaMemcpy(h_U, U, m*m*sizeof(float), cudaMemcpyDeviceToHost));
    printf("\nU (%d*%d):\n",m,m);
    int i;
    for(i=0;i<m*m;i++)
        printf("%.3f ", h_U[i]);

    CHECK(cudaMallocHost(&h_S,m*sizeof(float)));
    CHECK(cudaMemcpy(h_S, S, m*sizeof(float), cudaMemcpyDeviceToHost));
    printf("\nS (%d):\n",m);
    for(i=0;i<m;i++)
        printf("%.3f ", h_S[i]);

    CHECK(cudaMallocHost(&h_V_T,n*n*sizeof(float)));
    CHECK(cudaMemcpy(h_V_T, V_T, n*n*sizeof(float), cudaMemcpyDeviceToHost));
    printf("\nV_T (%d*%d):\n",n,n);
    for(i=0;i<n*n;i++)
        printf("%.3f ", h_V_T[i]);

    //END DEBUG

}

int main(){

    int n,k,m,i,j;
    n = 10;
    k = 5;
    m = n*k;

    srand(time(NULL));

    float theta[n*m];
    //Fill theta with random values between 0 and 1
    for(i=0; i<n*m; i++)
        theta[i] = rand()/(float)RAND_MAX;

    float alpha[m];
    //Fill optimal solution alpha with k random values
    memset(alpha,0,m*sizeof(float));
    for(i=0; i<k; i++){
        j = rand()%m;
        if(alpha[j] != 0)
            i--;
        else
            alpha[j] = rand()/(float)RAND_MAX;
    }

    //DEBUG
    printf("theta:\n");
    printf("[");
    for(i=0; i<n*m; i++){
        if(i%m==0)
            printf("[%.3f,",theta[i]);
        else if(i == n*m-1)
            printf("%.3f]",theta[i]);
        else if((i+1)%m==0)
            printf("%.3f],",theta[i]);
        else
            printf("%.3f,",theta[i]);
    }
    printf("]\n");
    //END DEBUG

    //MoorePenroseInverse
    float *d_theta,*d_thetaPseudoInv;
    CHECK(cudaMalloc(&d_theta, n*m*sizeof(float)));
    CHECK(cudaMemcpy(d_theta, theta, n*m*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_thetaPseudoInv, m*n*sizeof(float)));
    MoorePenroseInverse(d_theta, n, m, d_thetaPseudoInv);

    //Calbulate b = theta * alpha

    //call k_LiMapS

    //Check result

    return 0;

}
