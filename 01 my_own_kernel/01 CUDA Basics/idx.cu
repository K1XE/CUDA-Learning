#include <stdio.h>
#include <bits/stdc++.h>

__global__ void whoAmI(void)
{
    // 不妨设 Grid -> (4, 4, 4)
    // 某个block位于 -> (0, 2, 3)
    // 最终block_id -> 0 + 2 * 4 + 3 * 4 * 4 = 56
    int block_id =                          // block的位置
        blockIdx.x +                        // 0
        blockIdx.y * gridDim.x +            // 2 * 4
        blockIdx.z * gridDim.x * gridDim.y; // 3 * 4 * 4

    int block_offset =
        block_id *                            // 第m个block
        blockDim.x * blockDim.y * blockDim.z; // Don't Forget the Thread

    // 同上 即将 Grid <-> Block => Block <-> Thread
    int thread_offset =
        threadIdx.x +
        threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y;

    // 当前线程id = block的偏移量 +
    //             此thread所在block内部thread的偏移量
    int global_thread_id = block_offset + thread_offset;

    // 32个 thread -> 1个 warp
    printf("%4d | Block(%d %d %d) = %4d | Thread(%d %d %d) = %4d\n",
           global_thread_id,
           blockIdx.x, blockIdx.y, blockIdx.z, block_id,
           threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
}

int main(int argc, char **argv)
{
    const int b_x = 2, b_y = 2, b_z = 2;
    const int t_x = 4, t_y = 4, t_z = 4;

    int block_per_grid   = b_x * b_y * b_z;
    int thread_per_block = t_x * t_y * t_z;

    std::cout << block_per_grid << "block/grid" << std::endl;
    std::cout << thread_per_block << "thread/block" << std::endl;

    dim3 blockPerGrid(b_x, b_y, b_z);
    dim3 threadPerBlock(t_x, t_y, t_z);

    whoAmI<<<blockPerGrid, threadPerBlock>>>();

    cudaDeviceSynchronize();

    return 0;
}