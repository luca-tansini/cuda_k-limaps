#include "k-LiMapS.cu"
#include <cusolverSp.h>

//Calcuates the MoorePenroseInverse matrix of the input matrix a, leaving the result in apseudoinv
void MoorePenroseInverse(float *a, int n, int m, float *apseudoinv){

    //Calculate theta SVD via cuSOLVER api
    cusolverDnHandle_t cusolverHandle;
    cusolverDnCreate(&cusolverHandle);
    //Get dimension needed for the workspace buffer
    int bufferDim;
    cusolverDnSgesvd_bufferSize(cusolverHandle, n, m, &bufferDim);

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

    //MoorePenroseInverse

    //Calbulate b = theta * alpha

    //call k_LiMapS

    //Check result

    return 0;

}
