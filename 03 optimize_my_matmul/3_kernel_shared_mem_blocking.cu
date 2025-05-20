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

    // tile的索引
    const uint block_row = blockIdx.x;
    const uint block_col = blockIdx.y;

    // tile内部elem的索引
    const uint inner_row = threadIdx.x / BLOCKSIZE;
    const uint inner_col = threadIdx.x % BLOCKSIZE;

    // 获取此线程需要计算的部分
    A += block_row * BLOCKSIZE * K; //
    B += block_col * BLOCKSIZE; //
    C += block_row * BLOCKSIZE * N + block_col * BLOCKSIZE;

    // sums 一开始初始化为 0，它就是用来“累加”那一行 A 和那一列 B 整个长度 K 上对应元素乘积的结果。
    float sums = 0.0f;

    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE)
    {
        AShared[inner_row * BLOCKSIZE + inner_col] = A[inner_row * K + inner_col];
        BShared[inner_row * BLOCKSIZE + inner_col] = B[inner_row * N + inner_col];
        
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

        // 移动到下一个tile
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        // 开始处理tile内部元素
        for (int inIdx = 0; inIdx < BLOCKSIZE; inIdx ++ )
        {
            sums += AShared[inner_row * BLOCKSIZE + inner_col] * BShared[inIdx * BLOCKSIZE + inner_col];
        }

        __syncthreads(); //
    }
    
    // 累积完所有 tile，再写回 C， 计算结果放入C的对应位置
    C[inner_row * N + inner_col] = alpha * sums + beta * C[inner_row * N + inner_col];
}                                       