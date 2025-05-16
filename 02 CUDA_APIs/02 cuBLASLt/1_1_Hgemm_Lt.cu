#include <cuda_runtime.h>
#include <cublasLt.h>
#include <iostream>
#include <vector>

// Dimensions m and k must be multiples of 4.
#define M 4
#define K 4
#define N 4

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
    

    float h_A[M * K] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f};
    float h_B[K * N] = {
        1.0f, 2.0f, 4.0f, 4.0f,
        7.0f, 7.0f, 7.0f, 7.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f};
    float h_C_cpu[M * N] = {0};
    float h_C_gpu[M * N] = {0};

    __half *d_A_fp16, *d_B_fp16, *d_C_fp16;

    CHECK_CUDA(cudaMalloc(&d_A_fp16, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B_fp16, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C_fp16, M * N * sizeof(half)));

    std::vector<half> h_A_half(M * K);
    std::vector<half> h_B_half(K * N);

    for (int i = 0; i < M * K; i++)
        h_A_half[i] = __float2half(h_A[i]);
    for (int i = 0; i < K * N; i++)
        h_B_half[i] = __float2half(h_B[i]);

    CHECK_CUDA(cudaMemcpy(d_A_fp16, h_A_half.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_fp16, h_B_half.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));

    cublasLtHandle_t handleLt1;
    CHECK_CUBLAS(cublasLtCreate(&handleLt1));
    
    // 用于描述矩阵的数据类型、维度和存储方式
    cublasLtMatrixLayout_t matA_fp16, matB_fp16, matC_fp16;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matA_fp16, CUDA_R_16F, K, M, K)); 
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matB_fp16, CUDA_R_16F, N, K, N)); 
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matC_fp16, CUDA_R_16F, N, M, N)); 

    // 运算描述符
    cublasLtMatmulDesc_t matmulDesc_fp16;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc_fp16, CUBLAS_COMPUTE_16F, CUDA_R_16F));

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc_fp16, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc_fp16, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t)));

    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);

    CHECK_CUBLAS(cublasLtMatmul(
        handleLt1,
        matmulDesc_fp16,
        &alpha,
        d_B_fp16, matB_fp16,
        d_A_fp16, matA_fp16,
        &beta,
        d_C_fp16, matC_fp16,
        d_C_fp16, matC_fp16,
        NULL, NULL,
        0, 0
    ));

    std::vector<__half> h_C_gpu_half(M * N);
    CHECK_CUDA(cudaMemcpy(h_C_gpu_half.data(), d_C_fp16, M * N * sizeof(__half), cudaMemcpyDeviceToHost));

    // 转换精度
    for (int i = 0; i < M * N; i ++ )
    {
        h_C_gpu[i] = __half2float(h_C_gpu_half[i]);
    }

    cpuMatMul(h_A, h_B, h_C_cpu);

    std::cout << "矩阵 A 为（" << M << "x" << K << "）：" << std::endl;
    PRINT_MATRIX(h_A, M, K);
    std::cout << "矩阵 B 为（" << K << "x" << N << "）：" << std::endl;
    PRINT_MATRIX(h_B, K, N);

    std::cout << "CPU 运算结果（" << M << "x" << N << "）：" << std::endl;
    PRINT_MATRIX(h_C_cpu, M, N);
    std::cout << "cuBLASLt 半精度通用矩阵乘法结果（" << M << "x" << N << "）：" << std::endl;
    PRINT_MATRIX(h_C_gpu, M, N);


    // 正确性测试
    bool correct_fp16 = 1;
    for (int i = 0; i < M * N; i ++ )
    {
        if (fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-2f)
        {
            correct_fp16 = 0;
            exit(EXIT_FAILURE);
        }
    }
    if (correct_fp16) std::cout << "运算误差在容忍度内！" << std::endl;

    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matA_fp16));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matB_fp16));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matC_fp16));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmulDesc_fp16));
    CHECK_CUBLAS(cublasLtDestroy(handleLt1));
    CHECK_CUDA(cudaFree(d_A_fp16));
    CHECK_CUDA(cudaFree(d_B_fp16));
    CHECK_CUDA(cudaFree(d_C_fp16));
    
    return 0;
}