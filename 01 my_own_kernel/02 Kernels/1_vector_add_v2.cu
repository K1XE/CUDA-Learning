#include<stdio.h>
#include<bits/stdc++.h>

#define N 10000000          // 容器内元素个数
#define BLOCK_SIZE_1D 1024  // 一维的单位Block内thread数量
#define BLOCK_SIZE_3D_X 16  // 三维的单位Block内<x>方向上thread数量
#define BLOCK_SIZE_3D_Y 8   // 三维的单位Block内<y>方向上thread数量
#define BLOCK_SIZE_3D_Z 8   // 三维的单位Block内<z>方向上thread数量
#define EPOCH 100           // 测试轮数

// 一维运算核函数
__global__ void vec_add_gpu_1d(float* a, float* b, float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

// 三维运算核函数
__global__ void vec_add_gpu_3d(float* a, float* b, float* c, int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x,
        j = blockIdx.y * blockDim.y + threadIdx.y,
        k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i < nx && j < ny && k < nz)
    {
        int idx = i + j * nx + k * nx * ny;
        if (idx < nx * ny * nz)
        {
            c[idx] = a[idx] + b[idx];
        }
    }
}

// host端运算
void vec_add_cpu(float* a, float* b, float* c, int n)
{
    for (int i = 0; i < n; i ++ )
    {
        c[i] = a[i] + b[i];
    }
}

// 获取当前时间
double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// 初始化容器值
void init_vec(float* vec, int n)
{
    for (int i = 0; i < n; i ++ )
    {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

int main(int argc, char** argv)
{
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu_1d, *h_c_gpu_3d;
    float *d_a, *d_b, *d_c_gpu_1d, *d_c_gpu_3d;
    size_t size = N * sizeof(float);

    // 为host端分配内存
    h_a        = (float*)malloc(size);
    h_b        = (float*)malloc(size);
    h_c_cpu    = (float*)malloc(size);
    h_c_gpu_1d = (float*)malloc(size);
    h_c_gpu_3d = (float*)malloc(size);

    // 初始化容器
    srand(41);
    init_vec(h_a, N);
    init_vec(h_b, N);

    // 为device端分配内存
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c_gpu_1d, size);
    cudaMalloc(&d_c_gpu_3d, size);

    // 容器元素拷贝至device端
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 定义一维 grid 内 block 数量
    //    一维 block 内 thread 数量
    int num_block_1d = (N + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;

    // 定义三维 grid 内 block 数量
    //    三维 block 内 thread 数量
    int nx = 100,
        ny = 100,
        nz = 1000;
    dim3 block_size_3d(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);
    dim3 num_block_3d(
        (nx + block_size_3d.x - 1) / block_size_3d.x,
        (ny + block_size_3d.y - 1) / block_size_3d.y,
        (nz + block_size_3d.z - 1) / block_size_3d.z
    );
    
    // 预热
    std::cout << "性能测试预热中..." << std::endl;
    for (int i = 0; i < 3; i ++ )
    {
        vec_add_cpu(h_a, h_b, h_c_cpu, N);
        vec_add_gpu_1d<<<num_block_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_gpu_1d, N);
        vec_add_gpu_3d<<<num_block_3d, block_size_3d>>>(d_a, d_b, d_c_gpu_3d, nx, ny, nz);
        cudaDeviceSynchronize();
    }

    // CPU 基准测试启动
    std::cout << "持续" << EPOCH << "轮 | CPU版本的基准测试进行中..." << std::endl;
    double cpu_total_time = 0.0;
    for (int i = 0; i < EPOCH; i ++ )
    {
        double start_time = get_time();
        vec_add_cpu(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / EPOCH;

    // GPU <1d> 基准测试启动
    std::cout << "持续" << EPOCH << "轮 | GPU 1D版本的基准测试进行中..." << std::endl;
    double gpu_1d_total_time = 0.0;
    for (int i = 0; i < EPOCH; i++)
    {
        double start_time = get_time();
        vec_add_gpu_1d<<<num_block_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_gpu_1d, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_1d_total_time += end_time - start_time;
    }
    double gpu_1d_avg_time = gpu_1d_total_time / EPOCH;

    // GPU <3d> 基准测试启动
    std::cout << "持续" << EPOCH << "轮 | GPU 3D版本的基准测试进行中..." << std::endl;
    double gpu_3d_total_time = 0.0;
    for (int i = 0; i < EPOCH; i++)
    {
        double start_time = get_time();
        vec_add_gpu_3d<<<num_block_3d, block_size_3d>>>(d_a, d_b, d_c_gpu_3d, nx, ny, nz);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_3d_total_time += end_time - start_time;
    }
    double gpu_3d_avg_time = gpu_3d_total_time / EPOCH;

    // 输出结果
    std::cout << "CPU 测试的平均时间为" << cpu_avg_time * 1000 << "毫秒" << std::endl;
    std::cout << "GPU 1D测试的平均时间为" << gpu_1d_avg_time * 1000 << "毫秒" << std::endl;
    std::cout << "GPU 3D测试的平均时间为" << gpu_3d_avg_time * 1000 << "毫秒" << std::endl;
    std::cout << "1D 已加速:" << cpu_avg_time / gpu_1d_avg_time << "x" << std::endl;
    std::cout << "3D 已加速:" << cpu_avg_time / gpu_3d_avg_time << "x" << std::endl;
    std::cout << "(1D vs. 3D)已加速:" << gpu_1d_avg_time / gpu_3d_avg_time << "x" << std::endl;

    // 验证准确性
    cudaMemcpy(h_c_gpu_1d, d_c_gpu_1d, size, cudaMemcpyDeviceToHost);
    bool correct_1d = 1;
    for (int i = 0; i < N; i ++ )
    {
        if (fabs(h_c_gpu_1d[i] - h_c_cpu[i]) > 1e-5)
        {
            correct_1d = 0;
            break;
        }
    }
    std::cout << "GPU 1D验证结果是" << (correct_1d ? "正确的" : "错误的") << std::endl;
    cudaMemcpy(h_c_gpu_3d, d_c_gpu_3d, size, cudaMemcpyDeviceToHost);
    bool correct_3d = 1;
    for (int i = 0; i < N; i ++ )
    {
        if (fabs(h_c_gpu_3d[i] - h_c_cpu[i]) > 1e-5)
        {
            correct_3d = 0;
            break;
        }
    }
    std::cout << "GPU 3D验证结果是" << (correct_3d ? "正确的" : "错误的") << std::endl;
    
    // 释放内存
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu_1d);
    free(h_c_gpu_3d);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_gpu_1d);
    cudaFree(d_c_gpu_3d);
    
    return 0;
}