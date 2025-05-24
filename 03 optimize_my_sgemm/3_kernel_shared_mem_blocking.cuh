#pragma once

#include <cuda_runtime.h>


// sums 存的到底是什么？

/*
循环刚开始时
float sums = 0.0f;
*/

/*
处理第一个 tile
// tile 0: k = 0 … B-1
for (t = 0; t < B; ++t)
    sums += AShared[row][t] * BShared[t][col];
这时
      B−1 
sums= ∑   A[…,t]×B[t,…].
      t=0
*/

/*
处理第二个 tile
// tile 1: k = B … 2B-1
for (t = 0; t < B; ++t)
    sums += AShared[row][t] * BShared[t][col];
这时新加载的 AShared 和 BShared 对应的是下一段 k=B 开始的元素，累加到之前的 sums：

       B−1                 B−1
sums=( ∑ A[…,t]×B[t,…] )+( ∑ A[…,B+t]×B[B+t,…]).
       t=0                 t=0
*/


/*
……继续下去，直到覆盖所有 tile。
*/



template <const uint BLOCKSIZE>
__global__ void sgemmSharedMemBlocking(const float *A, const float *B, float *C,
                                       float alpha, float beta,
                                       const uint M, const uint K, const uint N)
{
    __shared__ float AShared[BLOCKSIZE * BLOCKSIZE];
    __shared__ float BShared[BLOCKSIZE * BLOCKSIZE];

    // tile 的索引
    const uint block_row = blockIdx.x;
    const uint block_col = blockIdx.y;

    // tile 内部元素的索引
    const uint inner_row = threadIdx.x / BLOCKSIZE;
    const uint inner_col = threadIdx.x % BLOCKSIZE;

    // 全局坐标：当前线程负责计算的 C 的位置
    const uint global_row = block_row * BLOCKSIZE + inner_row;
    const uint global_col = block_col * BLOCKSIZE + inner_col;

    float sums = 0.0f;

    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE)
    {
        // ✅ 正确加载 A 与 B 的 tile（考虑 k 方向偏移）
        AShared[inner_row * BLOCKSIZE + inner_col] = A[global_row * K + bkIdx + inner_col];
        BShared[inner_row * BLOCKSIZE + inner_col] = B[(bkIdx + inner_row) * N + global_col];

        // 确保每个元素均被填充完毕
        /*
            Shared AShared:                   Shared BShared:
            ┌────────────────┐                ┌────────────────┐
            │ a00 a01 ...    │                │ b00 b01 ...    │
            │ a10 a11 ...    │                │ b10 b11 ...    │
            │ ...     ...    │                │ ...     ...    │
            └────────────────┘                └────────────────┘
        */
        __syncthreads();


        // 开始处理tile内部元素
        // 每个线程通过 sums += ... 计算 A 的一行和 B 的一列的乘积总和，最终得到 C 的一个元素，多个线程并行构成整个 C 矩阵。
        for (int t = 0; t < BLOCKSIZE; t++)
        {
            sums += AShared[inner_row * BLOCKSIZE + t] * BShared[t * BLOCKSIZE + inner_col];
        }

        __syncthreads();
    }

    // ✅ 写入 C 的对应元素
    C[global_row * N + global_col] = alpha * sums + beta * C[global_row * N + global_col];
}
