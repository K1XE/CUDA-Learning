#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cassert>

#include "3_kernel_shared_mem_blocking.cuh"

#define CHECK_CUDA(call)                                                         \
    do                                                                           \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess)                                                  \
        {                                                                        \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl;           \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

void matmul_cpu(const float *A, const float *B, float *C, float alpha, float beta,
                int M, int K, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

int main()
{
    const int M = 256, K = 256, N = 256;
    const int BLOCKSIZE = 16;
    const float alpha = 1.0f, beta = 0.0f;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_C_ref(M * N, 0.0f);

    for (int i = 0; i < M * K; i++)
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; i++)
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C.data(), size_C, cudaMemcpyHostToDevice));

    dim3 grid(M / BLOCKSIZE, N / BLOCKSIZE);
    dim3 block(BLOCKSIZE * BLOCKSIZE); // 保持核函数中的 threadIdx.x 使用逻辑

    // Warmup
    sgemmSharedMemBlocking<BLOCKSIZE><<<grid, block>>>(d_A, d_B, d_C, alpha, beta, M, K, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 10; i++)
    {
        sgemmSharedMemBlocking<BLOCKSIZE><<<grid, block>>>(d_A, d_B, d_C, alpha, beta, M, K, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= 10; // 平均每次 kernel 的耗时

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    matmul_cpu(h_A.data(), h_B.data(), h_C_ref.data(), alpha, beta, M, K, N);

    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++)
    {
        max_diff = std::max(max_diff, std::abs(h_C[i] - h_C_ref[i]));
    }

    float gflops = 2.0f * M * N * K / (ms * 1e6f);

    std::cout << "SharedMem Blocking Kernel elapsed time: " << ms << " ms\n";
    std::cout << "Max diff: " << max_diff << std::endl;
    std::cout << "Test " << (max_diff < 1e-4f ? "PASSED" : "FAILED") << std::endl;
    std::cout << "GFLOPS: " << gflops << std::endl;

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    return 0;
}
