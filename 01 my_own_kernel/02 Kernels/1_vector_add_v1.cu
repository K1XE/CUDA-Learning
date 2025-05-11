#include <bits/stdc++.h>
#include <stdio.h>
#include <time.h>
#define N 10000000
#define BLOCK_SIZE 256

// vector add CPU version
void vector_add_cpu(float *a, float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

// vector add GPU version
__global__ void vector_add_gpu(float *a, float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

// 对容器元素随机初始化
void init_vector(float *vec, int n)
{
    for (int i = 0; i < n; i++)
    {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

// 计时
double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    // 为host端分配内存
    h_a     = (float *)malloc(size);
    h_b     = (float *)malloc(size);
    h_c_cpu = (float *)malloc(size);
    h_c_gpu = (float *)malloc(size);

    // 初始化容器
    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    // 为device端分配内存
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 将数据从 host -> device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 定义Grid与Block维度 即Block个数与单位内Thread个数
    int num_block = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 预热
    std::cout << "性能测试预热中..." << std::endl;
    for (int i = 0; i < 3; i++)
    {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu<<<num_block, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }

    // CPU 基准测试启动
    int epoch = 20;
    std::cout << "持续" << epoch << "轮 | CPU版本的基准测试进行中..." << std::endl;
    double cpu_total_time = 0.0;
    for (int i = 0; i < epoch; i++)
    {
        double start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / epoch;

    // GPU 基准测试启动
    std::cout << "持续" << epoch << "轮 | GPU版本的基准测试进行中..." << std::endl;
    double gpu_total_time = 0.0;
    for (int i = 0; i < epoch; i++)
    {
        double start_time = get_time();
        vector_add_gpu<<<num_block, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / epoch;

    // 输出结果
    std::cout << "CPU 测试的平均时间为" << cpu_avg_time * 1000 << "毫秒" << std::endl;
    std::cout << "GPU 测试的平均时间为" << gpu_avg_time * 1000 << "毫秒" << std::endl;
    std::cout << "已加速" << cpu_avg_time / gpu_avg_time << "x" << std::endl;

    // 验证结果
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    bool correct = 1;
    for (int i = 0; i < N; i++)
    {
        if (fabs(h_c_gpu[i] - h_c_cpu[i]) > 1e-5)
        {
            correct = 0;
            break;
        }
    }
    std::cout << "验证结果是" << (correct ? "正确的" : "错误的") << std::endl;

    // Don't forget free memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}