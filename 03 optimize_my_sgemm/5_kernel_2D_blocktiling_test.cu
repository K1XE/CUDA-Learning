#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <assert.h>

#include "5_kernel_2D_blocktiling.cuh"

#define CHECK_CUDA(call)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// CPU 版本 SGEMM 供验证
void cpu_sgemm(const float *A, const float *B, float *C,
               float alpha, float beta, int M, int K, int N)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.f;
            for (int k = 0; k < K; ++k)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

int main()
{
    constexpr int M = 64;
    constexpr int K = 64;
    constexpr int N = 64;

    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 8;
    constexpr int TM = 4;
    constexpr int TN = 4;

    const float alpha = 1.f, beta = 0.f;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_C = (float *)malloc(sizeC);
    float *h_C_ref = (float *)malloc(sizeC);

    // 初始化随机数据
    for (int i = 0; i < M * K; ++i)
        h_A[i] = (rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; ++i)
        h_B[i] = (rand() % 100) / 100.0f;
    for (int i = 0; i < M * N; ++i)
        h_C[i] = h_C_ref[i] = 0.f;

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sizeA));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, sizeC));

    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 blockDim((BM * BN) / (TM * TN));

    // 事件用于计时
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    sgemm2DBlocktiling<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(
        d_A, d_B, d_C, alpha, beta, M, K, N);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    printf("2D v2 Kernel elapsed time: %.3f ms\n", elapsed_ms);

    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // CPU 参考实现
    cpu_sgemm(h_A, h_B, h_C_ref, alpha, beta, M, K, N);

    // 验证正确性
    float max_diff = 0.f;
    for (int i = 0; i < M * N; ++i)
    {
        float d = fabs(h_C[i] - h_C_ref[i]);
        if (d > max_diff)
            max_diff = d;
    }
    printf("Max diff: %f\n", max_diff);
    if (max_diff < 1e-3f)
        printf("Test PASSED.\n");
    else
        printf("Test FAILED.\n");

    float gflops = 2.f * M * N * K / (elapsed_ms * 1e-3f) / 1e9f;
    printf("2D v2 GFLOPS: %.2f\n", gflops);

    // 释放资源
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
