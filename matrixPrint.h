#include <stdio.h>

void printColumnMajorMatrix(float *A, int nrows, int ncols){
    int i,j;
    for(i=0; i<nrows; i++){
        for(j=0; j<ncols; j+=1)
            printf("%.3f ", A[j*nrows + i]);
        printf("\n");
    }
}

void printColumnMajorMatrixForPython(float *A, int nrows, int ncols){
    int i,j;
    printf("[");
    for(i=0; i<nrows; i++){
        printf("[");
        for(j=0; j<ncols; j+=1)
            if(j == ncols-1)
                printf("%.3f", A[j*nrows + i]);
            else
                printf("%.3f,", A[j*nrows + i]);
        if(i == nrows-1)
            printf("]");
        else
            printf("],");
    }
    printf("]\n\n");
}
