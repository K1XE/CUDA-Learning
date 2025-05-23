/*
 * sgemmVectorize.cu
 *
 * 简要说明：
 *  这是一个基于 CUDA 的单精度矩阵乘加（SGEMM）优化实现，
 *  计算形式：C = alpha * A * B + beta * C。
 *  主要优化点：
 *    1. 瓦片化(Block Tiling)：将输出矩阵划分为 BM×BN 大小的小块，由不同 block 计算。
 *    2. 共享内存缓存(Shared Memory)：先将 A、B 子块加载到共享内存 As、Bs，减少全局内存访问。
 *    3. 向量化加载(Vectorized Loads)：使用 float4 一次性加载 4 个 float，提高内存带宽利用率。
 *    4. A 子块即时转置(On-the-fly Transpose)：加载 A 时将 BM×BK 片转为 BK×BM，
 *       使后续按列访问共享内存时可以连续读取。
 *    5. 寄存器阻塞(Register Blocking)：每个线程负责计算 TM×TN 的小结果块，
 *       在寄存器中累加多个内积，最后一次性写回。
 */
#pragma once

#include <cuda_runtime.h>

// BM, BN, BK: block tile 大小
// TM, TN: 每个线程负责的子 tile 大小
template <const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__global__ void sgemmVectorize(
    const float *A, const float *B, float *C,
    float alpha, float beta,
    const uint M, const uint K, const uint N)
{
    // 一、计算这个 block 在 C 矩阵中的子块位置 (cRow, cCol)
    uint cRow = blockIdx.y;
    uint cCol = blockIdx.x;

    // 二、在 block 内，把所有线程按一维编号 threadIdx.x 从 0 开始，
    //    再划成 (BM/TM) 行 × (BN/TN) 列 的虚拟线程网格
    uint threadRow = threadIdx.x / (BN / TN); // 行号 [0 .. BM/TM)
    uint threadCol = threadIdx.x % (BN / TN); // 列号 [0 .. BN/TN)

    // 三、给这个 block 分配共享内存，用于缓存 A 的 BM×BK 子块，B 的 BK×BN 子块
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 四、偏移全局指针到本 block 负责的子块起点
    //    A: 跳到第 cRow*BM 行，第 0 列
    //    B: 跳到第 0 行，第 cCol*BN 列
    //    C: 跳到第 cRow*BM 行，第 cCol*BN 列
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // 五、每个线程一次用 float4 读 4 个 float。下面算它们在子块中要读的位置：
    uint numLoadsA = (BM * BK) / 4; // A 子块里总共有多少组 float4
    uint numLoadsB = (BK * BN) / 4; // B 子块里总共有多少组 float4

    // 在 A 子块里：按行主序平铺，划分成 numLoadsA 个 float4，
    // innerRowA = 哪一组的「行索引（除以每行 load 数量）」
    // innerColA = 在那一行里的第几个 float4
    uint innerRowA = threadIdx.x / (BK / 4);
    uint innerColA = threadIdx.x % (BK / 4);

    // 同理在 B 子块里
    uint innerRowB = threadIdx.x / (BN / 4);
    uint innerColB = threadIdx.x % (BN / 4);

    // 六、为每个线程在寄存器中分配：
    //   threadRes: 长度 TM×TN，用来累加这一线程负责的子子块
    //   regM, regN: 各长度 TM、TN，用于在内积循环里做外积
    float threadRes[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    // 七、按 K 方向分段加载和计算
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        // —— 1）向量化读 A 并瞬时转置 存入共享
        // 全局 A 坐标: 行 = innerRowA, 列 = bkIdx + innerColA*4 .. +3
        float4 a_vec = reinterpret_cast<const float4 *>(&A[innerRowA * K + innerColA * 4])[0];
        // —— 向量化读 A 并瞬时转置 存入共享 As ——
        // 假设当前子块 A 的形状是 BM 行 × BK 列。
        // 共享内存 As 被当作一个 [BK × BM] 的矩阵来存，即 As[row, col] 对应 As[row*BM + col]。

        // 这 4 个元素 (a_vec.x, .y, .z, .w) 来自原矩阵 A 子块的同一行、4 列连续位置：
        //    global_row = cRow*BM + innerRowA
        //    global_col = bkIdx + innerColA*4 + d,  d = 0,1,2,3
        //
        // 在 A 子块里的坐标：(innerRowA, innerColA*4 + d)
        //
        // 我们希望把它们“瞬时转置”到 As 中：
        //    As[ new_row = (innerColA*4 + d), new_col = innerRowA ]
        //
        // 这样，原来 A 子块的一行 -> As 的一列。便于后续按 dotIdx 从 As 连续读「一行」时，
        // 实际在物理内存里是沿着 As 中的行（也即原 A 的列）按连续地址访问。

        // —— 可视化示意 ——
        //
        //   A 子块 (BM×BK)                存到 As (BK×BM)
        //   ┌─────────────────────┐       ┌─────────────────────┐
        //   │  a00 a01 a02 a03 …  │       │  a00 a10 a20 …      │    ← As 的第 0 行对应 A 的第 0 列
        //   │  a10 a11 a12 a13 …  │       │  a01 a11 a21 …      │    ← As 的第 1 行对应 A 的第 1 列
        //   │  a20 a21 a22 a23 …  │  -->  │  a02 a12 a22 …      │
        //   │  a30 a31 a32 a33 …  │       │  a03 a13 a23 …      │    ← 一次读 4 列，展开到 As 的 4 行
        //   │     …               │       │  …                  │
        //   └─────────────────────┘       └─────────────────────┘
        //
        // 具体到这 4 个元素：
        //   (innerRowA, innerColA*4 + 0) => As[ row = (innerColA*4 + 0), col = innerRowA ]
        //   (innerRowA, innerColA*4 + 1) => As[ row = (innerColA*4 + 1), col = innerRowA ]
        //   (innerRowA, innerColA*4 + 2) => As[ row = (innerColA*4 + 2), col = innerRowA ]
        //   (innerRowA, innerColA*4 + 3) => As[ row = (innerColA*4 + 3), col = innerRowA ]

        As[(innerColA * 4 + 0) * BM + innerRowA] = a_vec.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = a_vec.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = a_vec.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = a_vec.w;

        // —— 2）向量化读 B 存入共享（不转置）
        float4 b_vec = reinterpret_cast<const float4 *>(&B[innerRowB * N + innerColB * 4])[0];
        reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] = b_vec;

        // —— 确保所有线程都加载完
        __syncthreads();

        // —— 3）推进全局指针到下一个 BK 段
        A += BK;     // A 跳到同一行的下一个子块起点
        B += BK * N; // B 跳到下一个子块起点

        // —— 4）在寄存器里做内积累加
        // 我们之前把 A 子块（BM×BK）瞬时转置到 As（看作 BK 行 × BM 列）中，
        // 即 As[r, c] 存的是原 A 子块的 [c, r]。
        // 现在内层循环的 d 表示「在这一个 BK×BN 子块里，
        // 我们当前要用的第 d 列」（对应原 A 子块的第 d 行）。
        for (uint d = 0; d < BK; ++d)
        {
            // 从 As 的「第 d 列」（原子块的第 d 行）取 TM 元
            // 对于这一列，我们要取出 TM 个元素，构成一个小向量 regM，
            // 用来和 B 的相应向量做外积累加到 threadRes。
            //
            // threadRow 表示当前线程在 BM/TM 个「线程子行」里的序号，
            // 所以它在 As 的第几列负责：
            //    col = threadRow*TM + i,  i ∈ [0,TM)
            //
            // 完整的 As 二维索引：
            //    行 = d,
            //    列 = threadRow*TM + i
            //
            // 内存偏移（线性索引）就是
            //    d * BM   +   (threadRow*TM + i)
            //
            for (uint i = 0; i < TM; ++i)
                regM[i] = As[d * BM + threadRow * TM + i];

            // 从 Bs 的「第 d 行」（原子块的第 d 行）取 TN 元
            for (uint j = 0; j < TN; ++j)
                regN[j] = Bs[d * BN + threadCol * TN + j];

            // TM×TN 外积累加到 threadRes
            for (uint i = 0; i < TM; ++i)
                for (uint j = 0; j < TN; ++j)
                    threadRes[i * TN + j] += regM[i] * regN[j];
        }

        // —— 5）等所有线程计算完这一段，再开始加载下一块
        __syncthreads();
    }

    // 八、把累加结果写回 C (α·AB + β·C)
    for (uint i = 0; i < TM; ++i)
    {
        for (uint j = 0; j < TN; j += 4)
        {
            // 输出子矩阵里这一线程负责的第 (i,j) 组 4 元素
            int row = threadRow * TM + i;
            int col = threadCol * TN + j;

            // 一次读 4 元素
            float4 c_old = reinterpret_cast<const float4 *>(&C[row * N + col])[0];

            // 分别更新 .x,.y,.z,.w
            c_old.x = alpha * threadRes[i * TN + j + 0] + beta * c_old.x;
            c_old.y = alpha * threadRes[i * TN + j + 1] + beta * c_old.y;
            c_old.z = alpha * threadRes[i * TN + j + 2] + beta * c_old.z;
            c_old.w = alpha * threadRes[i * TN + j + 3] + beta * c_old.w;

            // 写回
            reinterpret_cast<float4 *>(&C[row * N + col])[0] = c_old;
        }
    }
}
