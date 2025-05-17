#include <cuda_runtime.h>
#include <cublasXt.h>
#include <iostream>
#include <vector>

#define M 256
#define K 256
#define N 256

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

void cpuMatMul(float *A, float *B, float *C, int m, int k, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            float sums = 0.0f;
            for (int l = 0; l < k; l++)
            {
                sums += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sums;
        }
    }
}

void init_mat(std::vector<float> &mat, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
    {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

int main(int argc, char **argv)
{
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_cpu(M * N);
    std::vector<float> h_C_gpu(M * N);

    init_mat(h_A, M, K);
    init_mat(h_B, K, N);

    cpuMatMul(h_A.data(), h_B.data(), h_C_cpu.data(), M, K, N);

    cublasXtHandle_t handleXt1;
    CHECK_CUBLAS(cublasXtCreate(&handleXt1));

    int device[1] = {0};
    CHECK_CUBLAS(cublasXtDeviceSelect(handleXt1, 1, device));

    float alpha = 1.0f;
    float beta = 0.0f;
    CHECK_CUBLAS(cublasXtSgemm(handleXt1,
                               CUBLAS_OP_N, CUBLAS_OP_N,
                               N, M, K,
                               &alpha,
                               h_B.data(), N,
                               h_A.data(), K,
                               &beta,
                               h_C_gpu.data(), N));
    bool correct_xt = 1;
    for (int i = 0; i < M * N; i ++ )
    {
        if (fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-4f)
        {
            correct_xt = 0;
        }
    }
    std::cout << (correct_xt ? "运算误差在容忍度内！" : "运算误差超出容忍度！") << std::endl;

    CHECK_CUBLAS(cublasXtDestroy(handleXt1));
    
    return 0;
}