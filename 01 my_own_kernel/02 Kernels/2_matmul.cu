#include <stdio.h>
#include <bits/stdc++.h>

#define M 256
#define K 512
#define N 256
#define BLOCK_SIZE 32
#define EPOCH 20

// cpu朴素矩阵乘法
void cpu_matmul(float *A, float *B, float *C, int m, int k, int n)
{
    for (int i = 0; i < m; i ++ )
    {
        for (int j = 0; j < n; j ++ )
        {
            float sums = 0.0f;
            for (int l = 0; l < k; l ++ )
            {
                sums += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sums;
        }
    }
}

// gpu版本
__global__ void gpu_matmul(float *A, float *B, float *C, int m, int k, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y,
        col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n)
    {
        float sums = 0.0f;
        for (int l = 0; l < k; l ++ )
        {
            sums += A[row * k + l] * B[l * n + col];
        }
        C[row * n + col] = sums;
    }
}

// 初始化矩阵
void init_mat(float *mat, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i ++ )
    {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// 获取当前时间
double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv)
{
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    float *d_A, *d_B, *d_C;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    h_A     = (float*)malloc(sizeA);
    h_B     = (float*)malloc(sizeB);
    h_C_cpu = (float*)malloc(sizeC);
    h_C_gpu = (float*)malloc(sizeC);

    init_mat(h_A, M, K);
    init_mat(h_B, K, N);

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    // 计算行数 要用到一列的thread数量 反之亦然
    // y 
    // ^
    // .
    // ·
    // ·   MATRIX
    // ·
    // ·
    // · · · · · · > x
    dim3 num_block(
        (N + block_size.x - 1) / block_size.x,
        (M + block_size.y - 1) / block_size.y
    );

    std::cout << "性能测试预热中..." << std::endl;
    for (int i = 0; i < 3; i ++ )
    {
        cpu_matmul(h_A, h_B, h_C_cpu, M, K, N);
        gpu_matmul<<<num_block, block_size>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
    }

    std::cout << "持续" << EPOCH << "轮 | CPU版本的基准测试进行中..." << std::endl;
    double cpu_total_time = 0.0;
    for (int i = 0; i < EPOCH; i ++ )
    {
        double start_time = get_time();
        cpu_matmul(h_A, h_B, h_C_cpu, M, K, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / EPOCH;

    std::cout << "持续" << EPOCH << "轮 | GPU版本的基准测试进行中..." << std::endl;
    double gpu_total_time = 0.0;
    for (int i = 0; i < EPOCH; i++)
    {
        double start_time = get_time();
        gpu_matmul<<<num_block, block_size>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / EPOCH;

    std::cout << "CPU 测试的平均时间为" << cpu_avg_time * 1e6f << "微秒" << std::endl;
    std::cout << "GPU 测试的平均时间为" << gpu_avg_time * 1e6f << "微秒" << std::endl;
    std::cout << "已加速:" << cpu_avg_time / gpu_avg_time << "x" << std::endl;

    // 浮点误差累计远超向量加法 增大误差容忍度
    cudaMemcpy(h_C_gpu, d_C, sizeC, cudaMemcpyDeviceToHost);
    bool correct = 1;
    for (int i = 0; i < M * N; i ++ )
    {
        if (fabs(h_C_gpu[i] - h_C_cpu[i]) > 1e-3f)
        {
            correct ^= correct;
            break;
        }
    }
    std::cout << "GPU验证结果是" << (correct ? "正确的" : "错误的") << std::endl;

    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}