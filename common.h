#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef _COMMON_H
#define _COMMON_H

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

#define CHECK_CUBLAS(call)                                                     \
{                                                                              \
    cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUSOLVER(call)                                                    \
{                                                                               \
    cusolverStatus_t err;                                                       \
    if ((err = (call)) != CUSOLVER_STATUS_SUCCESS)                              \
    {                                                                           \
        fprintf(stderr, "Got CUSOLVER error %d at %s:%d\n", err, __FILE__,      \
                __LINE__);                                                      \
        switch (err) {                                                          \
            case CUSOLVER_STATUS_NOT_INITIALIZED:                               \
                printf("CUSOLVER_STATUS_NOT_INITIALIZED\n");                    \
                break;                                                          \
            case CUSOLVER_STATUS_ALLOC_FAILED:                                  \
                printf("CUSOLVER_STATUS_ALLOC_FAILED\n");                       \
                break;                                                          \
            case CUSOLVER_STATUS_INVALID_VALUE:                                 \
                printf("CUSOLVER_STATUS_INVALID_VALUE\n");                      \
                break;                                                          \
            case CUSOLVER_STATUS_ARCH_MISMATCH:                                 \
                printf("CUSOLVER_STATUS_ARCH_MISMATCH\n");                      \
                break;                                                          \
            case CUSOLVER_STATUS_EXECUTION_FAILED:                              \
                printf("CUSOLVER_STATUS_EXECUTION_FAILED\n");                   \
                break;                                                          \
            case CUSOLVER_STATUS_INTERNAL_ERROR:                                \
                printf("CUSOLVER_STATUS_INTERNAL_ERROR\n");                     \
                break;                                                          \
            case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:                     \
                printf("CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED\n");          \
                break;                                                          \
        }                                                                       \
        /*exit(1);*/                                                                \
    }                                                                           \
}

#define CHECK_CURAND(call)                                                     \
{                                                                              \
    curandStatus_t err;                                                        \
    if ((err = (call)) != CURAND_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUFFT(call)                                                      \
{                                                                              \
    cufftResult err;                                                           \
    if ( (err = (call)) != CUFFT_SUCCESS)                                      \
    {                                                                          \
        fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__,        \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUSPARSE(call)                                                   \
{                                                                              \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
    {                                                                          \
        fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
        cudaError_t cuda_err = cudaGetLastError();                             \
        if (cuda_err != cudaSuccess)                                           \
        {                                                                      \
            fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                    cudaGetErrorString(cuda_err));                             \
        }                                                                      \
        exit(1);                                                               \
    }                                                                          \
}

inline void device_name() {
	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));
}

double seconds(){
	struct timeval tp;
	struct timezone tzp;
	gettimeofday(&tp, &tzp);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

#endif
