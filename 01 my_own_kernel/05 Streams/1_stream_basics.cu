#include <cuda_runtime.h>
#include <stdio.h>
#include <bits/stdc++.h>

#define N 100000
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file_name, const int lines)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error at <file_name>: " << file_name << " <line>: "
                  << lines << " code=" << static_cast<unsigned int>(err)
                  << " (" << cudaGetErrorString(err) << ") "
                  << " \"" << func << "\" " << std::endl;
    }
}

__global__ void vecAddKernel(float *a, float *b, float *c, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

void init_vec(float *vec, int n)
{
    for (int i = 0; i < n; i ++)
    {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

int main(int argc, char** argv)
{
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    // 声明流
    cudaStream_t stream1, stream2;

    // 创建流
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));

    // 预分配内存
    size_t size = N * sizeof(float);
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    // 初始化容器
    init_vec(h_a, N);
    init_vec(h_b, N);

    // device端
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c, size));

    // host 2 device 异步
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream2));

    // kernel execution
    vecAddKernel<<<(N + 255) / 256, 256, 0, stream1>>>(d_a, d_b, d_c, N);

    // device 2 host 异步
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_c, d_c, size, cudaMemcpyDeviceToHost, stream1));
    
    // 两个流 同时完成 同步
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));

    // 验证结果
    for (int i = 0; i < N; i ++ )
    {
        if (fabs(h_a[i] + h_b[i] - h_c[i]) > 1e-5)
        {
            std::cerr << "验证结果错误！" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    
    std::cout << "验证通过！" << std::endl;

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // 释放内存
    free(h_a);
    free(h_b);
    free(h_c);
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));

    return 0;
}
