#pragma once

#include <cuda_runtime.h>

__global__ void sgemmNaiveKernel(const float *A, const float *B, float *C,
                                 float alpha, float beta,
                                 int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sums = 0.0f;
        for (int i = 0; i < K; i ++ )
        {
            sums += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * sums + beta * C[row * N + col];
    }
}