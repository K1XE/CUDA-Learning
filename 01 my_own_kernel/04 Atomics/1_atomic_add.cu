#include <cuda_runtime.h>
#include <bits/stdc++.h>

#define BLOCK_SIZE 1000
#define NUM_BLOCK 1000

// 非原子操作的核函数 
__global__ void incrementCounterNonAtomic(int *counter)
{
    int old = *counter;
    int new_val = old + 1;
    *counter = new_val;
}

// 原子操作的核函数
__global__ void incrementCounterAtomic(int *counter)
{
    atomicAdd(counter, 1);
}

int main(int argc, char** argv)
{
    int h_counterNonAtomic = 0,
        h_counterAtomic = 0;
    int *d_counterNonAtomic,
        *d_counterAtomic;
    
    cudaMalloc((void**)&d_counterAtomic, sizeof(int));
    cudaMalloc((void**)&d_counterNonAtomic, sizeof(int));

    cudaMemcpy(d_counterAtomic, &h_counterAtomic, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counterNonAtomic, &h_counterNonAtomic, sizeof(int), cudaMemcpyHostToDevice);

    incrementCounterAtomic<<<NUM_BLOCK, BLOCK_SIZE>>>(d_counterAtomic);
    incrementCounterNonAtomic<<<NUM_BLOCK, BLOCK_SIZE>>>(d_counterNonAtomic);

    cudaMemcpy(&h_counterAtomic, d_counterAtomic, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_counterNonAtomic, d_counterNonAtomic, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "原子操作的自增函数最终结果为：" << h_counterAtomic << std::endl;
    std::cout << "非原子操作的自增函数最终结果为："  << h_counterNonAtomic << std::endl;

    cudaFree(d_counterAtomic);
    cudaFree(d_counterNonAtomic);

    return 0;
}