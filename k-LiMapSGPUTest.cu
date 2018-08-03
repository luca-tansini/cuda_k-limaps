#include "k-LiMapS.cu"
#include "MoorePenrose.cu"

/*
This main tests the k-LiMapS algorithm generating the dictionary and its pseudoinverse in GPU.
*/
int main(int argc, char **argv){

    if(argc != 2){
        printf("usage: k-LiMapSGPUTest n\n");
        exit(2);
    }

    int n,i,j;
    n = atoi(argv[1]);

    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    float cualpha = 1, cubeta = 0;

    float *d_b,*b;
    CHECK(cudaMalloc(&d_b, n*sizeof(float)));
    CHECK(cudaMallocHost(&b, n*sizeof(float)));

    for(int m = n; m <= 5*n; m+=n){

        printf("m = %d\n", m);

        //Generate dictionary theta with uniform random values between 0 and 1
        srand(time(NULL));
        float *theta;
        CHECK(cudaMallocHost(&theta, n*m*sizeof(float)));
        for(i=0; i<n*m; i++)
            theta[i] = rand()/(float)RAND_MAX;

        //calculate theta Moore-Penrose inverse
        float *d_theta,*d_thetaPseudoInv;
        CHECK(cudaMalloc(&d_theta, n*m*sizeof(float)));
        CHECK(cudaMemcpy(d_theta, theta, n*m*sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMalloc(&d_thetaPseudoInv, m*n*sizeof(float)));
        TransposedMoorePenroseInverse(d_theta, n, m, d_thetaPseudoInv);
        if(!CheckPseudoInverse(d_theta, n, m, d_thetaPseudoInv)){
            printf("something went wrong with the pseudoinverse!\n");
            return -2;
        }


        //transfer d_thetaPseudoInv into host memory
        float *thetaPseudoInv;
        CHECK(cudaMallocHost(&thetaPseudoInv, m*n*sizeof(float)));
        CHECK(cudaMemcpy(thetaPseudoInv, d_thetaPseudoInv, m*n*sizeof(float), cudaMemcpyDeviceToHost));

        //Allocate host and device alpha vectors
        float *alpha, *d_alpha, *limapsAlpha;
        CHECK(cudaMallocHost(&alpha, m*sizeof(float)));
        CHECK(cudaMalloc(&d_alpha, m*sizeof(float)));
        CHECK(cudaMallocHost(&limapsAlpha, m*sizeof(float)));

        for(int k = ceil(n/10); k <= ceil(n/2); k+= ceil(n/10)){

            int succ = 0;

            for(int iter= 0; iter < 100; iter++){

                //Fill optimal solution alpha with k random values
                memset(alpha,0,m*sizeof(float));
                for(i=0; i<k; i++){
                    j = rand()%m;
                    if(alpha[j] != 0)
                        i--;
                    else
                        alpha[j] = rand()/(float)RAND_MAX;
                }

                CHECK(cudaMemcpy(d_alpha, alpha, m*sizeof(float), cudaMemcpyHostToDevice));

                //Calculate b = theta * alpha
                CHECK_CUBLAS(cublasSgemv(cublasHandle, CUBLAS_OP_N, n, m, &cualpha, d_theta, n, d_alpha, 1, &cubeta, d_b, 1));
                CHECK(cudaMemcpy(b, d_b, n*sizeof(float), cudaMemcpyDeviceToHost));

                //call k_LiMapS
                k_LiMapS(k, theta, n, m, thetaPseudoInv, b, limapsAlpha, 1000);

                //Check result
                for(i=0; i<m; i++)
                    if(abs(alpha[i] - limapsAlpha[i]) > 1e-4)
                        break;
                if(i == m)
                    succ += 0;
            }

            printf("    k = %d --> %d%%\n", k, succ);

        }

        CHECK(cudaFree(d_theta));
        CHECK(cudaFree(d_thetaPseudoInv));
        CHECK(cudaFreeHost(theta));
        CHECK(cudaFreeHost(thetaPseudoInv));
        CHECK(cudaFree(d_alpha));
        CHECK(cudaFreeHost(alpha));
        CHECK(cudaFreeHost(limapsAlpha));

    }

    cudaDeviceReset();
    return 0;

}
