#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <assert.h>

#include "4_kernel_1D_blocktiling.cuh"
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

// CPU 版本 SGEMM（仅用于较小尺寸验证）
void cpu_sgemm(const float *A, const float *B, float *C,
               float alpha, float beta, int M, int K, int N)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.f;
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

int main()
{
    // 不同规模下做对比
    std::vector<int> sizes = {64, 256, 512, 1024};
    // tile 参数保持一致
    constexpr int BM = 32, BN = 32, BK = 8, TM = 4, TN = 4;
    const float alpha = 1.f, beta = 0.f;
    const int repeat = 5;

    // 预热 CUDA 上下文
    cudaFree(nullptr);

    printf(" size | 1D time(ms) | 1D GFLOPS | 2D time(ms) | 2D GFLOPS | max_diff\n");
    printf("------+-------------+-----------+-------------+-----------+---------\n");

    for (int sz : sizes)
    {
        int M = sz, K = sz, N = sz;
        size_t nA = size_t(M) * K, nB = size_t(K) * N, nC = size_t(M) * N;
        size_t bytesA = nA * sizeof(float);
        size_t bytesB = nB * sizeof(float);
        size_t bytesC = nC * sizeof(float);

        std::vector<float> h_A(nA), h_B(nB), h_C(nC), h_C_ref;
        for (size_t i = 0; i < nA; ++i)
            h_A[i] = float(rand() % 100) / 100.f;
        for (size_t i = 0; i < nB; ++i)
            h_B[i] = float(rand() % 100) / 100.f;
        std::fill(h_C.begin(), h_C.end(), 0.f);
        if (sz <= 256)
            h_C_ref = h_C; // 仅对较小尺寸验证

        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, bytesA));
        CHECK_CUDA(cudaMalloc(&d_B, bytesB));
        CHECK_CUDA(cudaMalloc(&d_C, bytesC));
        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytesB, cudaMemcpyHostToDevice));

        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 blk1((BM / TM) * BN);
        dim3 blk2((BM * BN) / (TM * TN));

        float t1_sum = 0.f, t2_sum = 0.f;
        for (int r = 0; r < repeat; ++r)
        {
            CHECK_CUDA(cudaMemset(d_C, 0, bytesC));
            cudaEvent_t s1, e1;
            CHECK_CUDA(cudaEventCreate(&s1));
            CHECK_CUDA(cudaEventCreate(&e1));
            CHECK_CUDA(cudaEventRecord(s1));
            sgemm1DBlocktiling<BM, BN, BK, TM><<<grid, blk1>>>(d_A, d_B, d_C, alpha, beta, M, K, N);
            CHECK_CUDA(cudaEventRecord(e1));
            CHECK_CUDA(cudaEventSynchronize(e1));
            float t1;
            CHECK_CUDA(cudaEventElapsedTime(&t1, s1, e1));
            t1_sum += t1;
            CHECK_CUDA(cudaEventDestroy(s1));
            CHECK_CUDA(cudaEventDestroy(e1));

            CHECK_CUDA(cudaMemset(d_C, 0, bytesC));
            cudaEvent_t s2, e2;
            CHECK_CUDA(cudaEventCreate(&s2));
            CHECK_CUDA(cudaEventCreate(&e2));
            CHECK_CUDA(cudaEventRecord(s2));
            sgemm2DBlocktiling<BM, BN, BK, TM, TN><<<grid, blk2>>>(d_A, d_B, d_C, alpha, beta, M, K, N);
            CHECK_CUDA(cudaEventRecord(e2));
            CHECK_CUDA(cudaEventSynchronize(e2));
            float t2;
            CHECK_CUDA(cudaEventElapsedTime(&t2, s2, e2));
            t2_sum += t2;
            CHECK_CUDA(cudaEventDestroy(s2));
            CHECK_CUDA(cudaEventDestroy(e2));
        }
        float t1_avg = t1_sum / repeat;
        float t2_avg = t2_sum / repeat;
        float gflops1 = 2.f * M * N * K / (t1_avg * 1e-3f) / 1e9f;
        float gflops2 = 2.f * M * N * K / (t2_avg * 1e-3f) / 1e9f;

        float max_diff = 0.f;
        if (sz <= 256)
        {
            CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, bytesC, cudaMemcpyDeviceToHost));
            cpu_sgemm(h_A.data(), h_B.data(), h_C_ref.data(), alpha, beta, M, K, N);
            for (size_t i = 0; i < nC; ++i)
                max_diff = fmaxf(max_diff, fabsf(h_C[i] - h_C_ref[i]));
        }

        printf("%4d | %11.3f | %9.2f | %11.3f | %9.2f | %8.6f\n",
               sz, t1_avg, gflops1, t2_avg, gflops2, max_diff);

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
    }

    return 0;
}
