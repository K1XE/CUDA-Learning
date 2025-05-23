/*
 * sgemmGlobalMemCoalesce.cu
 *
 * 简要说明：
 *  这是一个基于 CUDA 的单精度矩阵乘加（SGEMM）示例，
 *  计算公式：C = alpha * A * B + beta * C
 *
 *  主要优化点：
 *    1. 全局内存访问对齐(Coalesced Access)：
 *       通过合理映射 threadIdx.x 到 C 中的 (row, col)，
 *       使得同一个 warp 的线程能够访问 B 矩阵同一行的连续元素，
 *       从而实现内存事务合并。
 *    2. 线程映射(Thread Mapping)：
 *       每个线程负责计算 C 矩阵的一个元素，
 *       通过 threadIdx.x 的整除与取模运算将其映射到二维索引 (crow, ccol)。
 *    3. 简单内积(Scalar Reduction)：
 *       每个线程沿 K 方向做标量乘加，
 *       对 B 的访问是连续的，可被合并为少量内存事务。
 */
#pragma once

#include <cuda_runtime.h>

/*
与1_naive_matmul.cu区别：
// 每个线程访问一个 C[i][j]
// 内层循环：for (int k = 0; k < K; ++k)
// 访问 B[k][j] → 即 B 的列是固定的、行是变的

线程访问 B:
         B[0][j]
         B[1][j]
         B[2][j]
         B[3][j]

// 每个线程块处理一个 tile
// 每个线程处理 tile 中的一格
// 线程沿着 B 的某一行读取多个连续的值：B[i][0], B[i][1], B[i][2], B[i][3]

线程访问 B:
       B[i][0]   B[i][1]   B[i][2]   B[i][3]


*/

// 使访问矩阵 B 时的全局内存访问是 coalesced（合并的）
// 每个线程负责计算 C 中的一个元素，多个线程同时访问 B 中第 i 行的相邻列，形成连续的内存访问。

/*
⬛ 表示一个 threadIdx.x 的线程

矩阵 C (全局)，线程块 (blockIdx.x = 1, blockIdx.y = 2) 映射区域：

   ...   ...   ...   ...   ...   ...   ...   ...   8    9   10   11   ...
   ...   ...   ...   ...   ...   ...   ...   ...  ---  ---  ---  ---  ← row 4
   ...   ...   ...   ...   ...   ...   ...   ...  ---  ---  ---  ---  ← row 5
   ...   ...   ...   ...   ...   ...   ...   ...  ---  ---  ---  ---  ← row 6
   ...   ...   ...   ...   ...   ...   ...   ...  ---  ---  ---  ---  ← row 7

                                                    ↑
                                            全局列号 ccol = 8 ~ 11
*/

/*
第一步：threadIdx.x 映射成 2D 坐标（块内局部）
threadIdx.x 的 1D 编号被展开为 2D 块内坐标：
(假设 BLOCKSIZE = 4)

threadIdx.x   →   (row_in_block, col_in_block)
     0        →         (0, 0)
     1        →         (0, 1)
     2        →         (0, 2)
     3        →         (0, 3)
     4        →         (1, 0)
     5        →         (1, 1)
     6        →         (1, 2)
     7        →         (1, 3)
     8        →         (2, 0)
     9        →         (2, 1)
    10        →         (2, 2)
    11        →         (2, 3)
    12        →         (3, 0)
    13        →         (3, 1)
    14        →         (3, 2)
    15        →         (3, 3)

这个映射由这两行公式完成：

row_in_block = threadIdx.x / BLOCKSIZE; // 整除
col_in_block = threadIdx.x % BLOCKSIZE; // 取模

第二步：加上 blockIdx，映射为全局坐标
假设：
blockIdx.x = 1
blockIdx.y = 2
BLOCKSIZE = 4

crow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
     =      1       *     4    +   row_in_block
     = 4 + row_in_block

ccol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
     =      2       *     4    +   col_in_block
     = 8 + col_in_block

*/

template <const uint BLOCKSIZE>
__global__ void sgemmGlobalMemCoalesce(const float *A, const float *B, float *C,
                                       float alpha, float beta,
                                       int M, int K, int N)
{
    int crow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    int ccol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (crow < M && ccol < N)
    {
        float sums = 0.0f;
        for (int i = 0; i < K; i++)
        {
            sums += A[crow * K + i] * B[i * N + ccol]; // 对于相同 i，多个线程（在一个 warp 中）访问不同的 ccol → 相邻列 → 连续地址
        }
        C[crow * N + ccol] = alpha * sums + beta * C[crow * N + ccol];
    }
}