#pragma once

#include <cuda_runtime.h>

template <const uint BM, const uint BN, const uint TM, const uint TN>
__global__ void sgemm(
    float *A, float *B, float *C,
    const float alpha, const float beta,
    const uint M, const uint K, const uint N)
{
    uint block_row = blockIdx.y;
    uint block_col = blockIdx.x;

    A += block_row * K * BM;
    B += block_col * BN;
    C += block_row * N * BM + block_col * BN;

    uint thread_row = threadIdx.x / (BN / TN);
    uint thread_col = threadIdx.x % (BN / TN);

    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];

    uint inner_row_A = threadIdx.x / (BK / 4);
    uint inner_col_A = threadIdx.x % (BK / 4);
    uint inner_row_B = threadIdx.x / (BN / 4);
    uint inner_col_B = threadIdx.x % (BN / 4);

    float threadRes[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        float4 A_tmp = reinterpret_cast<float4 *>(&A[inner_row_A * K + inner_col_A * 4])[0];
        As[(inner_col_A * 4 + 0) * BM + inner_row_A] = A_tmp.x;
        As[(inner_col_A * 4 + 1) * BM + inner_row_A] = A_tmp.y;
        As[(inner_col_A * 4 + 2) * BM + inner_row_A] = A_tmp.z;
        As[(inner_col_A * 4 + 3) * BM + inner_row_A] = A_tmp.w;
        reinterpret_cast<float4 *>(&Bs[inner_row_B * BN + inner_col_B * 4])[0] =
            reinterpret_cast<float4 *>(&B[inner_row_B * N + inner_col_B * 4])[0];

        __syncthreads();
        A += BK;
        B += BK * N;
        for (int inIdx = 0; inIdx < BK; inIdx++)
        {
            for (int i = 0; i < TM; i++)
            {
                regM[i] = As[inIdx * BM + thread_row * TM + i];
            }
            for (int j = 0; j < TN; j++)
            {
                regN[j] = Bs[inIdx * BN + thread_col * TN + j];
            }
            for (int i = 0; i < TM; i++)
            {
                for (int j = 0; j < TN; j++)
                {
                    threadRes[i * TN + j] += regM[i] * regN[j];
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; i++)
    {
        for (int j = 0; j < TN; j++)
        {
            int row = thread_row * TM + i;
            int col = thread_col * TN + i;
            float4 tmp = reinterpret_cast<float4 *>(&C[row * N + col])[0];

            tmp.x = alpha * threadRes[i * TN + j + 0] + beta * tmp.x;
            tmp.y = alpha * threadRes[i * TN + j + 1] + beta * tmp.y;
            tmp.z = alpha * threadRes[i * TN + j + 2] + beta * tmp.z;
            tmp.w = alpha * threadRes[i * TN + j + 3] + beta * tmp.w;

            reinterpret_cast<float4 *>(&C[row * N + col])[0] = tmp;
        }
    }
}