#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

#define M 3
#define K 4
#define N 2

#undef PRINT_MATRIX
#define PRINT_MATRIX(mat, rows, cols)              \
    for (int i = 0; i < rows; i++)                 \
    {                                              \
        for (int j = 0; j < cols; j++)             \
        {                                          \
            std::cout << mat[i * cols + j] << " "; \
        }                                          \
        std::cout << std::endl;                    \
    }

#define CHECK_CUDA(val)                                     \
    do                                                      \
    {                                                       \
        cudaError_t sta = val;                              \
        if (sta != cudaSuccess)                             \
        {                                                   \
            std::cerr << "CUDA异常，具体信息：" << __FILE__ \
                      << " " << __LINE__                    \
                      << " " << cudaGetErrorString(sta)     \
                      << std::endl;                         \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)

#define CHECK_CUBLAS(val)                                     \
    do                                                        \
    {                                                         \
        cublasStatus_t sta = val;                             \
        if (sta != CUBLAS_STATUS_SUCCESS)                     \
        {                                                     \
            std::cerr << "CUBLAS异常，具体信息：" << __FILE__ \
                      << " " << __LINE__                      \
                      << " " << sta                           \
                      << std::endl;                           \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

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
    float C_cpu[M * N], C_cubla_S[M * N];

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    float alpha = 1.0f, beta = 0.0f;

    cublasHandle_t handle1;
    CHECK_CUBLAS(cublasCreate(&handle1));
    CHECK_CUBLAS(cublasSgemm(
        handle1, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, N,
        d_A, K,
        &beta,
        d_C, N));

    /*  CHECK_CUBLAS(cublasSgemm(
        handle1, CUBLAS_OP_T, CUBLAS_OP_T,
        M, N, K,
        &alpha,
        d_B, N,
        d_A, K,
        &beta,
        d_C, M));
    */

    CHECK_CUDA(cudaMemcpy(C_cubla_S, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    cpuMatMul(A, B, C_cpu);

    std::cout << "矩阵 A 为（" << M << "x" << K << "）：" << std::endl;
    PRINT_MATRIX(A, M, K);
    std::cout << "矩阵 B 为（" << K << "x" << N << "）：" << std::endl;
    PRINT_MATRIX(B, K, N);

    std::cout << "CPU 运算结果（" << M << "x" << N << "）：" << std::endl;
    PRINT_MATRIX(C_cpu, M, N);
    std::cout << "cuBLAS 单精度通用矩阵乘法结果（" << M << "x" << N << "）：" << std::endl;
    PRINT_MATRIX(C_cubla_S, M, N);

    CHECK_CUBLAS(cublasDestroy(handle1));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}