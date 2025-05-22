#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "4_kernel_1D_blocktiling.cuh"

#define CHECK_CUDA(call)                                                                 \
    do                                                                                   \
    {                                                                                    \
        cudaError_t err = call;                                                          \
        if (err != cudaSuccess)                                                          \
        {                                                                                \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl;                           \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while (0)

void cpu_sgemm(const float *A, const float *B, float *C,
               float alpha, float beta, int M, int K, int N)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
}

int main()
{
    constexpr int M = 128, K = 128, N = 128;
    constexpr int BM = 64, BN = 64, BK = 8, TM = 4;

    int sizeA = M * K;
    int sizeB = K * N;
    int sizeC = M * N;

    std::vector<float> h_A(sizeA), h_B(sizeB), h_C(sizeC, 0), h_C_ref(sizeC, 0);

    for (int i = 0; i < sizeA; i++)
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < sizeB; i++)
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sizeA * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C.data(), sizeC * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block((BN));
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm1DBlocktiling<BM, BN, BK, TM><<<grid, block>>>(
        d_A, d_B, d_C, 1.0f, 0.0f, M, K, N);

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    // reference
    cpu_sgemm(h_A.data(), h_B.data(), h_C_ref.data(), 1.0f, 0.0f, M, K, N);

    // compare
    float max_diff = 0.0f;
    for (int i = 0; i < sizeC; i++)
    {
        float diff = std::fabs(h_C[i] - h_C_ref[i]);
        max_diff = std::max(max_diff, diff);
    }

    std::cout << "Max diff: " << max_diff << std::endl;
    if (max_diff < 1e-3f)
        std::cout << "Test PASSED." << std::endl;
    else
        std::cout << "Test FAILED." << std::endl;

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
