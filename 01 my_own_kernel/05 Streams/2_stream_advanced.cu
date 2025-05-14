#include <cuda_runtime.h>
#include <bits/stdc++.h>
#include <nvtx3/nvToolsExt.h>

#define BLOCK_SIZE 16
#define TILE_SIZE 16
#define M 256
#define K 512
#define N 256


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


// 优化mm
__global__ void matMulKernelOptimized(float *A, float *B, float *C, int m, int k, int n)
{
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sums = 0.0f;
    for (int tile = 0; tile < (k + TILE_SIZE - 1) / TILE_SIZE; tile ++ )
    {
        if (row < m && tile * TILE_SIZE + tx < k)
        {
            sharedA[ty][tx] = A[row * k + tile * TILE_SIZE + tx];
        }
        else sharedA[ty][tx] = 0.0f;

        if (col < n && tile * TILE_SIZE + ty < k)
        {
            sharedB[ty][tx] = B[(tile * TILE_SIZE + ty) * n + col];
        }
        else sharedB[ty][tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i ++ )
        {
            sums += sharedA[ty][i] * sharedB[i][tx];
        }

        __syncthreads();
    }
    if (row < m && col < n)
    {
        C[row * n + col] = sums;
    }
}

// 朴素mm
__global__ void matMulKernelNaive(float *A, float *B, float *C, int m, int k, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
    {
        float sums = 0.0f;

        for (int i = 0; i < k; i ++ )
        {
            sums += A[row * k + i] * B[i * n + col];
        }

        C[row * n + col] = sums;
    }
}

void cpuMatMul(float *A, float *B, float *C, int m, int k, int n)
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

void init_mat(float *mat, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i ++ )
    {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

void CUDART_CB streamCallBack(cudaStream_t stream, cudaError_t err, void *usrData)
{
    printf("流回调: 操作已完成\n");
}

int main(int argc, char** argv)
{
    nvtxRangePush("测试开始");

    float *h_A, *h_B, *h_C_gpu, *h_C_cpu;
    float *d_A, *d_B, *d_C;
    cudaStream_t stream1, stream2;
    cudaEvent_t event1;

    std::cout << event1 << std::endl;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    nvtxRangePush("主机内存分配");
    cudaMallocHost(&h_A, sizeA);
    cudaMallocHost(&h_B, sizeB);
    cudaMallocHost(&h_C_gpu, sizeC);
    cudaMallocHost(&h_C_cpu, sizeC);
    nvtxRangePop();

    nvtxRangePush("初始化矩阵");
    init_mat(h_A, M, K);
    init_mat(h_B, K, N);
    nvtxRangePop();

    // 保存结果用于验证
    cpuMatMul(h_A, h_B, h_C_cpu, M, K, N);

    nvtxRangePush("设备内存分配");
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    nvtxRangePop();

    nvtxRangePush("指定流的优先级");
    //查询当前设备所支持的 流优先级范围
    int leastPriority, greatestPriority;
    CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    
    // cudaStreamNonBlocking 表示这个流不与默认流（stream 0）隐式同步，任务可以并发执行。
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, leastPriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, greatestPriority));
    nvtxRangePop();

    // 创建事件
    CHECK_CUDA_ERROR(cudaEventCreate(&event1));
    
    // H2D 并执行核函数 （异步）
    nvtxRangePush("主机到设备的内存拷贝");
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
    nvtxRangePop();

    dim3 block_size_opt(TILE_SIZE, TILE_SIZE);
    dim3 num_block_opt(
        (N + block_size_opt.y - 1) / block_size_opt.y,
        (M + block_size_opt.x - 1) / block_size_opt.x
    );

    nvtxRangePush("stream1执行优化矩阵乘法核函数");
    matMulKernelOptimized<<<num_block_opt, block_size_opt, 0, stream1>>>(d_A, d_B, d_C, M, K, N);
    nvtxRangePop();

    // 给stream1打一个标签
    nvtxRangePush("在stream1中记录事件");
    CHECK_CUDA_ERROR(cudaEventRecord(event1, stream1));
    nvtxRangePop();

    // stream2等待事件完成
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream2, event1, 0));

    dim3 block_size_naive(BLOCK_SIZE, BLOCK_SIZE);
    dim3 num_block_naive(
        (N + block_size_naive.y - 1) / block_size_naive.y,
        (M + block_size_naive.x - 1) / block_size_naive.x
    );

    // 朴素矩阵乘法 覆盖d_C
    nvtxRangePush("stream2执行朴素矩阵乘法核函数");
    matMulKernelNaive<<<num_block_naive, block_size_naive, 0, stream2>>>(d_A, d_B, d_C, M, K, N);
    nvtxRangePop();

    // 在stream2中增加一个回调函数
    nvtxRangePush("在stream2中增加一个回调函数");
    CHECK_CUDA_ERROR(cudaStreamAddCallback(stream2, streamCallBack, NULL, 0));
    nvtxRangePop();

    // D2H
    nvtxRangePush("设备到主机的内存拷贝");
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_C_gpu, d_C, sizeC, cudaMemcpyDeviceToHost, stream2));
    nvtxRangePop();

    // 同步流
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));

    // 验证准确性
    for (int i = 0; i < M * N; i ++ )
    {
        if (fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-3f)
        {
            std::cerr << "验证结果在第" << i << "个元素错误！" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "验证通过！" << std::endl;

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 释放内存
    CHECK_CUDA_ERROR(cudaFreeHost(h_A));
    CHECK_CUDA_ERROR(cudaFreeHost(h_B));
    CHECK_CUDA_ERROR(cudaFreeHost(h_C_cpu));
    CHECK_CUDA_ERROR(cudaFreeHost(h_C_gpu));
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));
    CHECK_CUDA_ERROR(cudaEventDestroy(event1));

    nvtxRangePop();

    return 0;

}