/*
 * sgemmNaiveKernel.cu
 *
 * 简要说明：
 *  这是一个最朴素的单精度矩阵乘加（SGEMM）CUDA实现，
 *  计算公式：C = alpha * A * B + beta * C
 *  
 *  特点及不足：
 *    1. 直接映射：每个线程负责计算 C 矩阵中的一个元素 (row, col)。
 *    2. 全局内存访问非优化：
 *       访问 A 时同一 warp 连续线程读取的是 A 的不同行的同一列，
 *       访问 B 时同一 warp 连续线程读取的是 B 的同一列的不同行，
 *       导致内存访问无法合并，效率较低。
 *    3. 适合入门理解，与优化版本对比可见各种加速策略的必要性。
 */

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