#include "vectorUtility.cu"

#define BLOCK_SIZE 256

/*
Funzione che genera il dizionario D
I valori sono estratti dalla distribuzione gaussiana (media 0, stdev 1.0)
Ogni colonna è normalizzata (dividendo ogni elemento per la norma della colonna)
*/
void createDict(double *D, int n, int m){

    int blocksperdict = ceil(n*m*1.0/BLOCK_SIZE);
    int blockspercol = ceil(n*1.0/BLOCK_SIZE);

    srand(time(NULL));
    int seed = rand();

    curandState *devStates;
    CHECK(cudaMalloc((void **)&devStates, blocksperdict*BLOCK_SIZE*sizeof(curandState)));

    normfill<<<blocksperdict,BLOCK_SIZE>>>(D, n*m, devStates, seed);
    CHECK(cudaDeviceSynchronize());

    double *tmpcol;
    CHECK(cudaMalloc(&tmpcol, blockspercol*BLOCK_SIZE*sizeof(double)));

    for(int i=0; i<m; i++){

        //Copia per il calcolo della norma
        CHECK(cudaMemcpy(tmpcol, &D[i*n], n*sizeof(double), cudaMemcpyDeviceToDevice));

        //Calcola norma
        double norm = vectorNorm(tmpcol,n);

        //Chiama kernel che divide ogni elemento della colonna per la norma
        if(norm != 0){
            divide<<<blockspercol,BLOCK_SIZE>>>(&D[i*n], norm, n);
            CHECK(cudaDeviceSynchronize());
        }
    }
}

/*
Funzione che genera il vettore k-sparso alpha
I k valori sono estratti da una distribuzione normale (media 0, stdev 1.0)
*/
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
