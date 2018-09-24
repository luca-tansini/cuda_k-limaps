#include <stdio.h>

#define _MATRIX_PRINT_H

#define ANSI_COLOR_RED   "\x1b[31m"
#define ANSI_COLOR_RESET "\x1b[0m"

void printColumnMajorMatrix(double *A, int nrows, int ncols){
    int i,j;
    for(i=0; i<nrows; i++){
        for(j=0; j<ncols; j+=1)
            printf("%.5f ", A[j*nrows + i]);
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
                printf("%.5f", A[j*nrows + i]);
            else
                printf("%.5f,", A[j*nrows + i]);
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
            printf(ANSI_COLOR_RED "%.5f " ANSI_COLOR_RESET, v[i]);
        else
            printf("0.000 ");
    printf("\n");
}
