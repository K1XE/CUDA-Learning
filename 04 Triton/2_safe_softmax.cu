#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

#define BATCH_SIZE 32
#define NUMS 1024
#define CHECK_CUDA(val)                                     \
    do                                                      \
    {                                                       \
        cudaError_t sta = val;                              \
        if (sta != cudaSuccess)                             \
        {                                                   \
            std::cerr << "CUDA异常，具体信息：" << __FILE__ \
                      << " " << __LINE__                    \
                      << " " << cudaGetErrorString(sta)     \
                      << std::endl;                         \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)

template <const uint B, const uint N>
__global__ void safeSoftMaxKernel(float *mat, float *mat_process)
{
    int nid = threadIdx.x + blockIdx.x * blockDim.x;
    int bid = blockIdx.y;
    if (nid < N && bid < B)
    {
        uint offset = bid * N;
        float maxV = mat[offset];
        for (int i = 1; i < N; i++)
        {
            maxV = fmaxf(maxV, mat[offset + i]);
        }
        float sums = 0.0f;
        for (int i = 0; i < N; i++)
        {
            sums += expf(mat[offset + i] - maxV);
        }
        for (int i = 0; i < N; i++)
        {
            mat_process[offset + i] = expf(mat[offset + i] - maxV) / sums;
        }
    }
}

template <const uint B, const uint N>
void safeSoftmaxCpu(float *mat)
{
    for (int i = 0; i < B; i++)
    {
        float maxV = mat[i * N];
        for (int j = 1; j < N; j++)
        {
            maxV = max(maxV, mat[i * N + j]);
        }
        float sums = 0.0f;
        for (int j = 0; j < N; j++)
        {
            sums += exp(mat[i * N + j] - maxV);
        }
        for (int j = 0; j < N; j++)
        {
            mat[i * N + j] = exp(mat[i * N + j] - maxV) / sums;
        }
    }
}

int main(int argc, char **argv)
{
    std::vector<float> h_mat_cpu(BATCH_SIZE * NUMS);
    std::vector<float> h_mat_gpu(BATCH_SIZE * NUMS);

    size_t size = BATCH_SIZE * NUMS * sizeof(float);
    float *d_mat, *d_mat_process;

    // 初始化矩阵
    for (int i = 0; i < BATCH_SIZE * NUMS; i++)
    {
        h_mat_gpu[i] = h_mat_cpu[i] = (float)rand() / RAND_MAX;
    }

    CHECK_CUDA(cudaMalloc(&d_mat, size));
    CHECK_CUDA(cudaMalloc(&d_mat_process, size));

    CHECK_CUDA(cudaMemcpy(d_mat, h_mat_gpu.data(), size, cudaMemcpyHostToDevice));

    uint block = 256;
    uint grid_x = (NUMS + block - 1) / block;
    dim3 grid(grid_x, BATCH_SIZE);

    safeSoftMaxKernel<BATCH_SIZE, NUMS>
        <<<grid, block>>>(d_mat, d_mat_process);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_mat_gpu.data(), d_mat_process, size, cudaMemcpyDeviceToHost));

    safeSoftmaxCpu<BATCH_SIZE, NUMS>(h_mat_cpu.data());

    float max_diff = 0.0f;
    for (int i = 0; i < BATCH_SIZE * NUMS; i++)
    {
        float cur_diff = h_mat_cpu[i] - h_mat_gpu[i];
        max_diff = max(max_diff, cur_diff);
    }

    std::cout << "CPU与GPU版本Safe-Softmax最大误差为：" << max_diff << std::endl;

    CHECK_CUDA(cudaFree(d_mat));
    CHECK_CUDA(cudaFree(d_mat_process));

    return 0;
}