#include "k-LiMapS.cu"

/*
This main tests the k-LiMapS algorithm reading all the data needed for the test from standard input.
*/
int main(int argc, char **argv){

    int n,k,m,i;

    scanf("%d",&n);
    scanf("%d",&k);
    m = n*k;

    //Read dictionary theta
    float *theta;
    CHECK(cudaMallocHost(&theta, n*m*sizeof(float)));
    for(i=0; i<n*m; i++)
        scanf("%f",theta+i);

    //Read theta Moore-Penrose pseudoinverse
    float *thetaPseudoInv;
    CHECK(cudaMallocHost(&thetaPseudoInv, m*n*sizeof(float)));
    for(i=0; i<m*n; i++)
        scanf("%f",thetaPseudoInv+i);


    //Read optimal solution alpha
    float *alpha;
    CHECK(cudaMallocHost(&alpha, m*sizeof(float)));
    for(i=0; i<m; i++)
        scanf("%f",alpha+i);

    //Read signal b = theta * alpha (noiseless test)
    float *b;
    CHECK(cudaMallocHost(&b, n*sizeof(float)));
    for(i=0; i<n; i++)
        scanf("%f",b+i);

    //call k_LiMapS
    float *limapsAlpha;
    CHECK(cudaMallocHost(&limapsAlpha, m*sizeof(float)));

    k_LiMapS(k, theta, n, m, thetaPseudoInv, b, limapsAlpha, 1000);

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
    CHECK(cudaFreeHost(b));
    CHECK(cudaFreeHost(limapsAlpha));
    CHECK(cudaFreeHost(thetaPseudoInv));
    cudaDeviceReset();

    return 0;

}
