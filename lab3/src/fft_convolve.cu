/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_functions.h>
#include "fft_convolve.cuh"


/* 
Atomic-max function. You may find it useful for normalization.

We haven't really talked about this yet, but __device__ functions not
only are run on the GPU, but are called from within a kernel.

Source: 
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}



__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v, 
    cufftComplex *out_data,
    int padded_length) {


    /* TODO: Implement the point-wise multiplication and scaling for the
    FFT'd input and impulse response. 

    Recall that these are complex numbers, so you'll need to use the
    appropriate rule for multiplying them. 

    Also remember to scale by the padded length of the signal
    (see the notes for Question 1).

    As in Assignment 1 and Week 1, remember to make your implementation
    resilient to varying numbers of threads.

    */
    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    uint i = thread_index;
    // TODO: Update the while loop to handle all indices for this thread.
    //       Remember to advance the index as necessary.
    while (i<padded_length) {
        out_data[i].x = (raw_data[i].x * impulse_v[i].x - raw_data[i].y * impulse_v[i].y)/padded_length;
        out_data[i].y = (raw_data[i].x * impulse_v[i].y + raw_data[i].y * impulse_v[i].x)/padded_length;
        i += blockDim.x * gridDim.x;
    }
}

__global__
void
cudaMaximumKernel1(cufftComplex *out_data, float *max_abs_val,
                   int padded_length) {
    //calculate partial max for threads in the same block.
    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    float partial_max = 0.0;
    while (thread_index < padded_length) {
        partial_max = fmaxf(partial_max, fabs(out_data[thread_index].x));
        thread_index += blockDim.x * gridDim.x;
    }

    atomicMax(max_abs_val, partial_max);

}

__global__
void
cudaMaximumKernel2(cufftComplex *out_data, float *max_abs_val,
                   int padded_length) {
    //calculate partial max for threads in the same block.
    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float shmem[];
    float partial_max = 0.0f;

    while (thread_index < padded_length) {
        partial_max = fmaxf(partial_max, fabs(out_data[thread_index].x));
        thread_index += blockDim.x * gridDim.x;
    }

    shmem[threadIdx.x] = partial_max;
    __syncthreads();

    if (threadIdx.x==0){
        for(unsigned int threadIndex = 1; threadIndex < blockDim.x; ++threadIndex){
            partial_max = fmaxf(partial_max, shmem[threadIndex]);
        }

        atomicMax(max_abs_val, partial_max);
    }
}

__global__
void
cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the maximum-finding and subsequent
    normalization (dividing by maximum).

    There are many ways to do this reduction, and some methods
    have much better performance than others.
    */

    //calculate partial max for threads in the same block.
    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    float partial_max = 0.0f;
    extern __shared__ float partial_data[];
    while (thread_index < padded_length) {
        partial_max = fmaxf(partial_max, fabs(out_data[thread_index].x));
        thread_index += blockDim.x * gridDim.x;
    }
    unsigned int tid = threadIdx.x;
    partial_data[tid] = partial_max;
    __syncthreads();

    for(unsigned int s = blockDim.x/2; s>0; s>>=1){
        if(tid < s){
            partial_data[tid] = fmaxf(partial_data[tid], partial_data[tid+s]);
        }
        __syncthreads();
    }

    if (tid==0){
        atomicMax(max_abs_val, partial_data[0]);
    }
}

__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the division kernel. Divide all
    data by the value pointed to by max_abs_val. 

    This kernel should be quite short.
    */
    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    float max_allowed_amp = 0.99999;
    while (thread_index < padded_length) {
        out_data[thread_index].x *= (max_allowed_amp / *max_abs_val);
        thread_index += blockDim.x * gridDim.x;
    }
}


void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length) {

    cudaProdScaleKernel<<<blocks, threadsPerBlock>>>(raw_data, impulse_v, out_data, padded_length);
}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {

    cudaMaximumKernel<<<blocks, threadsPerBlock, threadsPerBlock* sizeof(float) >>>(out_data, max_abs_val, padded_length);

}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {

    cudaDivideKernel<<<blocks, threadsPerBlock, threadsPerBlock* sizeof(float) >>>(out_data, max_abs_val, padded_length);
}
