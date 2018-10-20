#include <stdio.h>

#define _MATRIX_PRINT_H

#define ANSI_COLOR_RED   "\x1b[31m"
#define ANSI_COLOR_RESET "\x1b[0m"

void printColumnMajorMatrix(double *A, int nrows, int ncols){
    int i,j;
    for(i=0; i<nrows; i++){
        for(j=0; j<ncols; j+=1)
            printf("%.7f ", A[j*nrows + i]);
        printf("\n");
    }
}

void printColumnMajorMatrixForPython(double *A, int nrows, int ncols){
    int i,j;
    printf("[");
    for(i=0; i<nrows; i++){
        printf("[");
        for(j=0; j<ncols; j+=1)
            if(j == ncols-1)
                printf("%.7f", A[j*nrows + i]);
            else
                printf("%.7f,", A[j*nrows + i]);
        if(i == nrows-1)
            printf("]");
        else
            printf("],");
    }
    printf("]\n\n");
}

void printColumnMajorMatrixForPythonWithPrecision(double *A, int nrows, int ncols, int precision){
    int i,j;
    if(precision < 1 || precision > 32) return;
    char s[8],s1[8];
    snprintf(s, 8, "%%.%df", precision);
    snprintf(s1, 8, "%%.%df,", precision);
    printf("[");
    for(i=0; i<nrows; i++){
        printf("[");
        for(j=0; j<ncols; j+=1)
            if(j == ncols-1)
                printf(s, A[j*nrows + i]);
            else
                printf(s1, A[j*nrows + i]);
        if(i == nrows-1)
            printf("]");
        else
            printf("],");
    }
    printf("]\n\n");
}

void printRowMajorMatrixForPython(double *A, int nrows, int ncols){
    int i,j;
    printf("[");
    for(i=0; i<nrows; i++){
        printf("[");
        for(j=0; j<ncols; j+=1)
            if(j == ncols-1)
                printf("%.7f", A[i*ncols + j]);
            else
                printf("%.7f,", A[i*ncols + j]);
        if(i == nrows-1)
            printf("]");
        else
            printf("],");
    }
    printf("]\n\n");
}

void printHighlightedVector(double *v, int len){
    int i;
    for(i=0; i<len; i++)
        if(v[i] != 0)
            printf(ANSI_COLOR_RED "%.7f " ANSI_COLOR_RESET, v[i]);
        else
            printf("0.000 ");
    printf("\n");
}

void printDeviceMatrix(double *d_A, int n, int m, const char *name){

    printf("%s [%dx%d]:\n", name, n, m);

    double *h_A;
    CHECK(cudaMallocHost(&h_A, n*m*sizeof(double)));
    CHECK(cudaMemcpy(h_A, d_A, n*m*sizeof(double), cudaMemcpyDeviceToHost));
    printColumnMajorMatrixForPython(h_A, n, m);
    printf("\n");

    CHECK(cudaFreeHost(h_A));
    return;

}
