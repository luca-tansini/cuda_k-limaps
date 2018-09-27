#include "k-LiMapS.cu"
#include "MoorePenrose.cu"
#include <curand.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 256

__global__ void normfill(double *D, int len, curandState *states, int seed){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < len){
        curand_init(tid*seed+seed, 0, 0, &states[tid]);
        D[tid] = curand_uniform_double(&states[tid]);
    }

}

__global__ void divide(double *v, double x, int len){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < len)
        v[tid] /= x;
}

//Function generating the dictionary
//The values are extracted from a normal distribution (mean 0, stdev 1.0)
//Then each column is forced to have norm == 1, dividing each element of the column by the column's norm
void createDict(double *D, int n, int m){

    int blocksperdict = ceil(n*m*1.0/BLOCK_SIZE);

    srand(time(NULL));
    int seed = rand();

    curandState *devStates;
    CHECK(cudaMalloc((void **)&devStates, blocksperdict*BLOCK_SIZE*sizeof(curandState)));

    normfill<<<blocksperdict,BLOCK_SIZE>>>(D, n*m, devStates, seed);
    CHECK(cudaDeviceSynchronize());

    double *tmpcol,*partialNormBlocks,norm;
    int blockspercol = ceil(n*1.0/BLOCK_SIZE);
    CHECK(cudaMalloc(&tmpcol, blockspercol*BLOCK_SIZE*sizeof(double)));
    CHECK(cudaMemset(tmpcol, 0, blockspercol*BLOCK_SIZE*sizeof(double)));
    CHECK(cudaMallocHost(&partialNormBlocks, blockspercol*sizeof(double)));

    for(int i=0; i<m; i++){

        CHECK(cudaMemcpy(tmpcol, &D[i*n], n*sizeof(double), cudaMemcpyDeviceToDevice));

        //CALCOLA NORMA CON vector2norm
        vector2norm<<<blockspercol,BLOCK_SIZE>>>(tmpcol);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(partialNormBlocks, tmpcol, blockspercol*sizeof(double), cudaMemcpyDeviceToHost));
        norm = 0;
        for(int j=0; j<blockspercol; j++)
            norm += partialNormBlocks[j];
        norm = sqrt(norm);

        //CHIAMA KERNEL CHE DIVIDE OGNI ELEMENTO PER LA NORMA
        divide<<<blockspercol,BLOCK_SIZE>>>(&D[i*n], norm, n);
        CHECK(cudaDeviceSynchronize());
    }
}

void generateAlpha(double *alpha, int m, int k){

    int blocksperk = ceil(k*1.0/BLOCK_SIZE);

    srand(time(NULL));
    int seed = rand();

    curandState *devStates;
    CHECK(cudaMalloc(&devStates, blocksperk*BLOCK_SIZE*sizeof(curandState)));

    double *d_kvalues,h_kvalues[k];
    CHECK(cudaMalloc(&d_kvalues, k*sizeof(double)));

    normfill<<<blocksperk,BLOCK_SIZE>>>(d_kvalues, k, devStates, seed);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_kvalues, d_kvalues, k*sizeof(double), cudaMemcpyDeviceToHost));

    double h_alpha[m];
    memset(h_alpha, 0, m*sizeof(double));
    for(int i=0; i<k; i++){
        int idx = rand()%m;
        if(h_alpha[idx] != 0)
            i--;
        else
            h_alpha[idx] = h_kvalues[i];
    }

    CHECK(cudaMemcpy(alpha, h_alpha, m * sizeof(double), cudaMemcpyHostToDevice));

}

int main(int argc, char **argv){

    if(argc != 2){
        printf("usage: noiselessTest <n>\n");
        return 2;
    }

    setbuf(stdout, NULL);

    int n = atoi(argv[1]);
    int m = 2*n;
    int k = n/4;

    //DEBUG
    printf("n:%d\tm:%d\tk:%d\n",n,m,k);
    //END DEBUG

    //CREA DIZIONARIO D
    printf("creating dictionary D...");
    double *D;
    CHECK(cudaMalloc(&D, n*m*sizeof(double)));

    createDict(D, n, m);
    printf("done\n");

    //CALCOLA PSEUDOINVERSA DINV
    printf("computing pseudoinverse DINV...");
    double *DINV;
    CHECK(cudaMalloc(&DINV, m*n*sizeof(double)));

    HostMoorePenroseInverse(D, n, m, DINV);

    if(!CheckPseudoInverse(D, n, m, DINV))
        printf("Something went wrong with the Moore-Penrose pseudoinverse!\n");

    printf("done\n");

    //GENERA ALPHAOPT
    printf("generating alphaopt...");
    double *alphaopt;
    CHECK(cudaMalloc(&alphaopt, m*sizeof(double)));

    generateAlpha(alphaopt, m, k);
    printf("done\n");

    //CALCOLA S = D * alphaopt
    printf("computing S = D * alphaopt...");
    double *s;
    CHECK(cudaMalloc(&s, n*sizeof(double)));

    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    double cualpha=1,cubeta=0;

    CHECK_CUBLAS(cublasDgemv(cublasHandle, CUBLAS_OP_N, n, m, &cualpha, D, n, alphaopt, 1, &cubeta, s, 1));
    printf("done\n");

    //CHIAMA K_LiMapS
    printf("calling K_LiMapS...");
    double *alphalimaps;
    CHECK(cudaMalloc(&alphalimaps, m*sizeof(double)));

    devMemK_LiMapS(k, D, n, m, DINV, s, alphalimaps, 1000);
    printf("done\n");

    //CHECK DEL RISULTATO
    double *h_alphalimaps,*h_alphaopt;
    CHECK(cudaMallocHost(&h_alphaopt, m*sizeof(double)));
    CHECK(cudaMallocHost(&h_alphalimaps, m*sizeof(double)));
    CHECK(cudaMemcpy(h_alphaopt, alphaopt, m*sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_alphalimaps, alphalimaps, m*sizeof(double), cudaMemcpyDeviceToHost));

    int i;
    for(i=0; i<m; i++)
        if(fabs(h_alphaopt[i] - h_alphalimaps[i]) > 1e-3)
            break;
    if(i == m)
        printf("ALL GOOD\n");
    else{
        printf("SOMETHING WENT WRONG!\n");

        printf("alphaopt:\n");
        printHighlightedVector(h_alphaopt, m);
        printf("\n");

        printf("alphalimaps:\n");
        printHighlightedVector(h_alphalimaps, m);
        printf("\n");
    }

    //FREE
    CHECK(cudaFree(D));
    CHECK(cudaFree(DINV));
    CHECK(cudaFree(alphaopt));
    CHECK(cudaFree(s));
    CHECK(cudaFree(alphalimaps));

    return 0;

}
