// 行主序（直觉） 与 列主序（cublas）

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

#define M 4
#define N 3
#define K 2
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char *const func_name, const char *const file_name, const int lines)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA于此文件发生错误：" << file_name << " 行数"
                  << lines << " 错误代码：" << static_cast<unsigned int>(err)
                  << " (" << cudaGetErrorString(err) << ") "
                  << " \"" << func_name << "\" " << std::endl;
    }
}

#define CHECK_CUBLAS_ERROR(val)                                            \
    {                                                                      \
        cublasStatus_t status = val;                                       \
        if (status != CUBLAS_STATUS_SUCCESS)                               \
        {                                                                  \
            std::cerr << "cuBLAS于此文件发生错误：" << __FILE__ << " 行数" \
                      << __LINE__ << " 错误代码：" << status << std::endl; \
        }                                                                  \
    }

int main(int argc, char **argv)
{
    float h_A[M * K] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f};
    float h_B[K * N] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f};
    float h_C[M * N];

    float *d_A, *d_B, *d_C;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    CHECK_CUDA_ERROR(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    cublasHandle_t handle1;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle1));

    float alpha = 1.0f, beta = 0.0f;

    // A (4, 2) B (2, 3) C (4, 3)
    // 要算 A * B = C 直觉理解三个矩阵均为行主序
    // 但cuBLAS默认列主序 即读取是为 A 与 B 的转置
    // 因此 我们求C的转置就是了
    // 最终得到的结果按cuBLAS来理解是C的转置（列主序）
    // 而我们理解则为需要的C矩阵（行主序）
    CHECK_CUBLAS_ERROR(cublasSgemm(handle1,
                                   CUBLAS_OP_N, CUBLAS_OP_N,
                                   N, M, K,                 // n, m, k — 实际计算的是 B × A
                                   &alpha,
                                   d_B, N,                  // B: K×N，列主序视角就是 N×K → 每列元素个数lda=N
                                   d_A, K,                  // A: M×K，列主序视角就是 K×M → 每列元素个数lda=K
                                   &beta,
                                   d_C, N));                // C: M×N，列主序就是 N×M → 每列元素个数ldc=N
    
    
    CHECK_CUDA_ERROR(cudaMemcpy(&h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
    
    std::cout << "C = A x B 的结果为：" << std::endl;
    for (int i = 0; i < M; i ++ )
    {
        for (int j = 0; j < N; j ++ )
        {
            std::cout << h_C[i * N + j] << "\t";
        }
        std::cout << std::endl;
    }
    
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUBLAS_ERROR(cublasDestroy(handle1));

    return 0;
}