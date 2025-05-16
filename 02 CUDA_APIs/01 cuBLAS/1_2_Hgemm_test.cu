#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define M 3
#define K 4
#define N 2

#define CHECK_CUDA(call)                                                                               \
    {                                                                                                  \
        cudaError_t err = call;                                                                        \
        if (err != cudaSuccess)                                                                        \
        {                                                                                              \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                              \
    }

#define CHECK_CUBLAS(call)                                                              \
    {                                                                                   \
        cublasStatus_t status = call;                                                   \
        if (status != CUBLAS_STATUS_SUCCESS)                                            \
        {                                                                               \
            fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE);                                                         \
        }                                                                               \
    }

#undef PRINT_MATRIX
#define PRINT_MATRIX(mat, rows, cols)            \
    for (int i = 0; i < rows; i++)               \
    {                                            \
        for (int j = 0; j < cols; j++)           \
            printf("%8.3f ", mat[i * cols + j]); \
        printf("\n");                            \
    }                                            \
    printf("\n");

void cpuMatMul(float *A, float *B, float *C)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sums = 0.0f;
            for (int l = 0; l < K; l++)
            {
                sums += A[i * K + l] * B[l * N + j];
            }
            C[i * N + j] = sums;
        }
    }
}

int main(int argc, char **argv)
{
    float A[M * K] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    float B[K * N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float C_cpu[M * N], C_cublas_h[M * N];

    cpuMatMul(A, B, C_cpu);

    cublasHandle_t handle1;
    CHECK_CUBLAS(cublasCreate(&handle1));

    __half *d_A_h, *d_B_h, *d_C_h;

    CHECK_CUDA(cudaMalloc(&d_A_h, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_B_h, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C_h, M * N * sizeof(__half)));

    // 精度转换
    __half A_h[M * K], B_h[K * N];
    for (int i = 0; i < M * K; i++)
    {
        A_h[i] = __float2half(A[i]);
    }
    for (int i = 0; i < K * N; i++)
    {
        B_h[i] = __float2half(B[i]);
    }

    CHECK_CUDA(cudaMemcpy(d_A_h, A_h, M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_h, B_h, K * N * sizeof(__half), cudaMemcpyHostToDevice));

    __half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);

    CHECK_CUBLAS(cublasHgemm(
        handle1,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha_h,
        d_B_h, N,
        d_A_h, K,
        &beta_h,
        d_C_h, N));

    __half C_h[M * N];
    CHECK_CUDA(cudaMemcpy(C_h, d_C_h, M * N * sizeof(__half), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N; i++)
    {
        C_cublas_h[i] = __half2float(C_h[i]);
    }

    printf("矩阵 A 为 (%dx%d):\n", M, K);
    PRINT_MATRIX(A, M, K);
    printf("矩阵 B 为 (%dx%d):\n", K, N);
    PRINT_MATRIX(B, K, N);

    printf("CPU 运算结果 (%dx%d):\n", M, N);
    PRINT_MATRIX(C_cpu, M, N);
    printf("cuBLAS 半精度通用矩阵乘法结果 (%dx%d):\n", M, N);
    PRINT_MATRIX(C_cublas_h, M, N);

    CHECK_CUDA(cudaFree(d_A_h));
    CHECK_CUDA(cudaFree(d_C_h));
    CHECK_CUDA(cudaFree(d_B_h));
    CHECK_CUBLAS(cublasDestroy(handle1));

    return 0;
}