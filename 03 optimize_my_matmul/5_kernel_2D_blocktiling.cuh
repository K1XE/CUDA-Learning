/*
 * sgemm2DBlocktiling.cu
 *
 * 简要说明：
 *  这是一个基于 CUDA 的单精度矩阵乘加（SGEMM）实现，
 *  计算公式：C = alpha * A * B + beta * C
 *
 *  主要优化点：
 *    1. 二维块划分(2D Block Tiling)：
 *       将输出矩阵分为 BM×BN 大小的 Block，每个 Block 由多个线程负责。
 *    2. 共享内存缓存(Shared Memory)：
 *       将 A 的 BM×BK 子块和 B 的 BK×BN 子块加载到 As、Bs，以减少全局内存访问。
 *    3. Thread 级别微分块运算(Register Blocking)：
 *       每个线程计算 TM×TN 大小的微输出块，将部分和保存在寄存器 threadRes[]。
 *    4. 均匀分配加载(Load Balancing)：
 *       线程通过 inner_row/inner_col 和 stride，将各自负责的行/列搬到共享内存，
 *       避免访问冲突并提升并行度。
 */
#pragma once

#include <cuda_runtime.h>

template <const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__global__ void sgemm2DBlocktiling(const float *A, const float *B, float *C,
                                   float alpha, float beta,
                                   const uint M, const uint K, const uint N)
{
    uint block_row = blockIdx.y;
    uint block_col = blockIdx.x;

    /*
        每个 thread 被赋予一个 tile（大小为 TM x TN），总共有 BM/TM x BN/TN 个小 tile
        thread_row: 当前 thread 分配到了哪个小 tile 的第几行
        thread_col: 当前 thread 分配到了哪个小 tile 的第几列
    */
    uint thread_row = threadIdx.x / (BN / TN);
    uint thread_col = threadIdx.x % (BN / TN);

    /*
    每个 block 要算 BM x BN 个输出元素（这是一个 tile）
    每个 thread 负责 TM x TN 个元素
    所以 thread 数量就是除法结果
    */
    uint oneBlockNeedCalcElemNums = BM * BN;
    uint oneThreadNeedCalcElemNums = TM * TN;
    uint threadNumsInOneBlock = oneBlockNeedCalcElemNums / oneThreadNeedCalcElemNums;

    assert(threadNumsInOneBlock > threadIdx.x);

    __shared__ float As[BM * BK + 8]; // pad 减少 bank conflict
    __shared__ float Bs[BK * BN + 8];

    // 不在循环中累加指针，用基址+偏移
    const float *A_tile = A + block_row * K * BM;
    const float *B_tile = B + block_col * BN;

    uint inner_row_A = threadIdx.x / BK; // 当前线程负责将 A tile 中哪一行搬到共享内存
    uint inner_col_A = threadIdx.x % BK; // 当前线程负责哪一列的元素

    // 行间跳跃步长（用于 for 循环搬多个元素）
    // 如果 thread 数是 64，stride_A = 64 / BK = 8，每个线程隔 8 行
    uint stride_A = threadNumsInOneBlock / BK;

    uint inner_row_B = threadIdx.x / BN;
    uint inner_col_B = threadIdx.x % BN;
    uint stride_B = threadNumsInOneBlock / BN;

    float threadRes[TM * TN] = {0.0f};
    float regM[TM] = {0.0f}; // 从共享内存读取 A 的某列（每个 thread 读 TM 个值，存到 regM[]）
    float regN[TN] = {0.0f}; // 从共享内存读取 B 的某行（每个 thread 读 TN 个值，存到 regN[]）

    /*
        A tile (BM x BK)                   B tile (BK x BN)

        ┌──────────────┐                   ┌────────────────┐
        │              │      *           │                │
        │  regM[TM]    │    dot product   │   regN[TN]     │
        │              │                  │                │
        └──────────────┘                   └────────────────┘

                    ↓ 每个 thread 计算 TM x TN 输出 tile
                threadRes[TM x TN] 累加更新
    */

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        // 每个线程可以搬运多个元素（以 stride 为步长），加载多个 row 的 A(B) 数据。
        for (int offset = 0; offset < BM; offset += stride_A)
        {
            int r = inner_row_A + offset;
            if (r < BM)
                As[r * BK + inner_col_A] = A_tile[r * K + bkIdx + inner_col_A];
        }
        for (int offset = 0; offset < BK; offset += stride_B)
        {
            int r = inner_row_B + offset;
            if (r < BK)
                Bs[r * BN + inner_col_B] = B_tile[(bkIdx + r) * N + inner_col_B];
        }

        __syncthreads();

        for (int inIdx = 0; inIdx < BK; inIdx++)
        {
            /*
                把 A 的一列（大小为 TM x 1）搬进 regM。
                当前线程负责的是一个 TM x TN 的输出 tile，需要 TM 个元素。
            */
            for (int i = 0; i < TM; i++)
            {
                regM[i] = As[(thread_row * TM + i) * BK + inIdx];
            }
            /*
                把 B 的一行（大小为 1 x TN）搬进 regN。
                当前线程负责 TN 列，需要 TN 个元素。
            */
            for (int j = 0; j < TN; j++)
            {
                regN[j] = Bs[inIdx * BN + thread_col * TN + j];
            }
            /*
                假设：
                TM = 2, TN = 3
                所以：每个线程负责一个 2 × 3 的输出块
                那么线程 t 要计算下面这些元素：
                C[row+0][col+0], C[row+0][col+1], C[row+0][col+2]
                C[row+1][col+0], C[row+1][col+1], C[row+1][col+2]
                此时 regM 和 regN 中的数据如下：
                regM[0] = A[row+0][k]
                regM[1] = A[row+1][k]

                regN[0] = B[k][col+0]
                regN[1] = B[k][col+1]
                regN[2] = B[k][col+2]
                其实threadRes[i * TN + j] += regM[i] * regN[j] 等价于 C[row+i][col+j] += A[row+i][k] * B[k][col+j];
            */

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


    /*
        由于每个线程负责计算 TM × TN 的小块（称为 tile）：
        因此输出 tile 的起始位置：
        行偏移：thread_row * TM
        列偏移：thread_col * TN
        (thread_row * TM + i) * N + thread_col * TN + j 等价于：
        row_idx = thread_row * TM + i;
        col_idx = thread_col * TN + j;
        C[row_idx][col_idx] = ...
    */


    for (int i = 0; i < TM; i++)
    {
        for (int j = 0; j < TN; j++)
        {
            int row = block_row * BM + thread_row * TM + i;
            int col = block_col * BN + thread_col * TN + j;
            if (row < M && col < N)
                // write into global C
                C[row * N + col] = alpha * threadRes[i * TN + j] + beta * C[row * N + col];
        }
    }
}
