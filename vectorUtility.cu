#ifndef _VECTOR_UTIL_CU_
#define _VECTOR_UTIL_CU_

//Kernel implementing: res = alpha * a + beta * b
__global__ void vectorSum(double alpha, double *a, double beta, double *b, double *res, int len){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < len)
        res[tid] = alpha * a[tid] + beta * b[tid];
}

//Kernel implementing the 2-norm of a vector (the vector is destroyed after computation, with v[i] being the partial sum of block i)
__global__ void vector2norm(double *v){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    v[tid] *= v[tid];

    int step = blockDim.x / 2;
    int idx = threadIdx.x;
    double *p = v + blockDim.x * blockIdx.x;
    while(step > 0){
        if(idx < step)
            p[idx] = p[idx] + p[idx+step];
        step /= 2;
        __syncthreads();
    }
    if(idx == 0)
        v[blockIdx.x] = p[idx];
}

#endif
