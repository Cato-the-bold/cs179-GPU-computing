#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "helper_cuda.h"
                                           
//helper_cuda.h contains the error checking macros. note that they're called
//CUDA_CALL and CUBLAS_CALL instead of the previous names


//TODO: perform the following matrix multiplications using cublas

#define M 2 // 
#define N 3
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

int main(int argc, char *argv[]) {
    using namespace std;

    float A[N * M] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    float B[M * N] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    float res1[N * N];
    float res2[M * M];
    float res1_[N * N];
    float res2_[M * M];

    int i, j, k;
    int cnt = M*N;
    float alpha = 1.0f;
    float beta = 0.0f;
    //TODO: cudaMalloc buffers, copy these to device, etc.
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);

    float *dev_A, *dev_B, *dev_C0, *dev_C1;
    cudaMalloc((float**)&dev_A, sizeof(float) * cnt);
    cudaMalloc((float**)&dev_B, sizeof(float) * cnt);
    cudaMalloc((float**)&dev_C0, sizeof(float) * N*N);
    cudaMalloc((float**)&dev_C1, sizeof(float) * M*M);

    status = cublasSetVector(cnt, sizeof(float), B, 1, dev_B, 1);
    status = cublasSetVector(cnt, sizeof(float), A, 1, dev_A, 1);

    // A * B
    // TODO: do this on GPU too with cuBLAS, copy result back, and printf it to check
    printf("A * B\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            res1[IDX2C(i, j, N)] = 0;
            for (k = 0; k < M; k++) {
                res1[IDX2C(i, j, N)] += A[IDX2C(i, k, N)] * B[IDX2C(k, j, M)];
            }
            printf("[%d, %d] = %f\n", i, j, res1[IDX2C(i, j, N)]);
        }
    }
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, M, &alpha, dev_A, N, dev_B, M, &beta, dev_C0, N);
    status = cublasGetVector(N*N, sizeof(float), dev_C0, 1, res1_, 1);

    if (std::equal(std::begin(res1), std::end(res1), std::begin(res1_)))
        cout << "Arrays are equal.";
    else
        cout << "Arrays are not equal.";

    // A^T * B^T
    // TODO: do this on GPU too with cuBLAS, copy result back, and printf to check it
    printf("\nA^T * B^T\n");
    for (i = 0; i < M; i++) {
        for (j = 0; j < M; j++) {
            res2[IDX2C(i, j, M)] = 0;
            for (k = 0; k < N; k++) {
                res2[IDX2C(i, j, M)] += A[IDX2C(k, i, N)] * B[IDX2C(j, k, M)];
            }
            printf("[%d, %d] = %f\n", i, j, res2[IDX2C(i, j, M)]);
        }
    }
    status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, M, N, &alpha, dev_A, N, dev_B, M, &beta, dev_C1, M);
    status = cublasGetVector(M*M, sizeof(float), dev_C0, 1, res2_, 1);

    if (std::equal(std::begin(res2), std::end(res2), std::begin(res2_)))
        cout << "Arrays are equal.";
    else
        cout << "Arrays are not equal."<<endl;

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C0);
    cudaFree(dev_C1);
    status = cublasDestroy(handle);
}