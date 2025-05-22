#pragma once

#include <cuda_runtime.h>

template <const uint BM, const uint BN, const uint BK, const uint TM>
__global__ void sgemm1DBlocktiling(const float *A, const float *B, float *C,
                                   float alpha, float beta,
                                   const uint M, const uint K, const uint N)
{

    // block_row: 当前 block 负责的 C 矩阵 tile 的行号（即处理第几块 BM 高度的 block）。
    // block_col: 当前 block 负责的 C 矩阵 tile 的列号（即处理第几块 BN 宽度的 block）。
    uint block_row = blockIdx.y;
    uint block_col = blockIdx.x;

    // 为了让每个线程知道它在 tile 中计算 C 的哪一列和哪一行组，
    // 用于 计算结果的寄存器缓存 threadRes[TM] 和写回 global memory 的地址定位
    // threadIdx.x 是 block 内的 1D thread index。
    // 每个线程负责 TM 行的计算
    uint thread_row = threadIdx.x / BN; // thread_row 表示负责 tile 中哪一“行组”
    uint thread_col = threadIdx.x % BN; // thread_col 表示负责 tile 中哪一“列”

    // 移动指针到当前 block tile 的位置
    const float *A_tile = A + block_row * BM * K;            // A 起始偏移
    const float *B_tile = B + block_col * BN;                // B 起始偏移

    // 减少全局内存读取开销，将 A 和 B 的 block tile 加载进共享内存中
    __shared__ float AShared[BM * BK];
    __shared__ float BShared[BK * BN];

    // 计算出 tile 中的一列 TM 个 C 值，存在 threadRes[0~TM-1] 中
    float threadRes[TM] = {0.0f};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) // 每个tile
    {
        // 让线程知道它该从 global memory 加载哪个元素到共享内存中
        uint inner_row_A = threadIdx.x / BK;
        uint inner_col_A = threadIdx.x % BK;
        uint inner_row_B = threadIdx.x / BN;
        uint inner_col_B = threadIdx.x % BN;

        // ---- 修正：用 A_tile 和 B_tile + bkIdx 做偏移，防止指针累加错位 ----
        if (inner_row_A < BM && inner_col_A < BK)
            AShared[inner_row_A * BK + inner_col_A] = A_tile[inner_row_A * K + bkIdx + inner_col_A];

        if (inner_row_B < BK && inner_col_B < BN)
            BShared[inner_row_B * BN + inner_col_B] = B_tile[(bkIdx + inner_row_B) * N + inner_col_B];

        __syncthreads();

        // 每个线程计算的是 C 的 同一列（thread_col），跨 TM 行 的那一列的值
        for (int inIdx = 0; inIdx < BK; inIdx++)
        {
            float tmpB = BShared[inIdx * BN + thread_col];
            for (int tdIdx = 0; tdIdx < TM; tdIdx++)
            {
                threadRes[tdIdx] += AShared[(thread_row * TM + tdIdx) * BK + inIdx] * tmpB;
            }
        }
        __syncthreads();
    }

    /*
     A block tile (BM x BK)         B block tile (BK x BN)
    +-------------------------+      +-------------------------+
    | A[i][k] A[i][k+1] ...   |      | B[k][j] B[k+1][j] ...   |
    | A[i+1][k] ...           |  x   |                         |
    | ...                    |      |                         |
    +-------------------------+      +-------------------------+

    =

    C[i][j], C[i+1][j], ..., C[i+TM-1][j]
    (由一个线程计算)
    */

    for (int tdIdx = 0; tdIdx < TM; tdIdx++)
    {
        // ---- 修正：写回全局C时使用绝对行列偏移 ----
        int row = block_row * BM + thread_row * TM + tdIdx;
        int col = block_col * BN + thread_col;
        if (row < M && col < N)
            C[row * N + col] = alpha * threadRes[tdIdx] + beta * C[row * N + col];
    }
}
