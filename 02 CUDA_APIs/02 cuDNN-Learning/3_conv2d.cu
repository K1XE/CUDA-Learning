#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>
#include <cmath>

#define WARMUP_EPOCH 10
#define BENCHMARK_EPOCH 100

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

#define CHECK_CUDNN(val)                                     \
    do                                                       \
    {                                                        \
        cudnnStatus_t sta = val;                             \
        if (sta != CUDNN_STATUS_SUCCESS)                     \
        {                                                    \
            std::cerr << "CUDNN异常，具体信息：" << __FILE__ \
                      << " " << __LINE__                     \
                      << " " << sta                          \
                      << std::endl;                          \
            exit(EXIT_FAILURE);                              \
        }                                                    \
    } while (0)

__global__ void naiveConv2d(float *input, float *kernel, float *output, int width, int height, int inChannels, int outChannels, int kernelSize, int batchSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int outChannel = blockIdx.z % outChannels;
    int batchIdx = blockIdx.z / outChannels;

    if (x < width && y < height && outChannel < outChannels && batchIdx < batchSize)
    {
        float sum = 0.0f;
        int halfKernel = kernelSize / 2;
        for (int inChannel = 0; inChannel < inChannels; inChannel++)
        {
            for (int ky = -halfKernel; ky <= halfKernel; ky++)
            {
                for (int kx = -halfKernel; kx <= halfKernel; kx++)
                {

                    /*
                    输入图（宽 x 高）：
                    +----+----+----+----+----+
                    |  1 |  2 |  3 |  4 |  5 |
                    +----+----+----+----+----+
                    |  6 |  7 | [8] |  9 | 10 |
                    +----+----+----+----+----+
                    | 11 | 12 | 13 | 14 | 15 |
                    +----+----+----+----+----+
                    | 16 | 17 | 18 | 19 | 20 |
                    +----+----+----+----+----+

                    卷积核（3x3）覆盖的区域：

                    (x=1,y=0)  (x=2,y=0)  (x=3,y=0)
                    +----+----+----+
                    |  2 |  3 |  4 |
                    +----+----+----+
                    |  7 | [8] | 9 |
                    +----+----+----+
                    | 12 | 13 | 14 |
                    +----+----+----+
                    */

                    // 找到输入图像的对应像素坐标
                    int ix = x + kx;
                    int iy = y + ky;
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height)
                    {
                        int inputIdx = ((batchIdx * inChannels + inChannel) * height + iy) * width + ix;
                        int kernelIdx = ((outChannel * inChannels + inChannel) * kernelSize + (ky + halfKernel)) * kernelSize + (kx + halfKernel);
                        sum += input[inputIdx] * kernel[kernelIdx];
                    }
                }
            }
        }
        int outputIdx = ((batchIdx * outChannels + outChannel) * height + y) * width + x;
        output[outputIdx] = sum;
    }
}

int main(int argc, char **argv)
{
    int bsz = 1;
    int in_channels = 1;
    int out_channels = 1; //
    int kernel_size = 3;
    int height = 4;
    int width = 4;
    int in_tensor_size = width * height * bsz * in_channels;
    int out_tensor_size = width * height * bsz * out_channels;
    int kernel_elements = kernel_size * kernel_size * in_channels * out_channels; //

    std::cout << "图像大小: " << width << "x" << height << "x" << in_channels << std::endl;
    std::cout << "卷积核大小: " << kernel_size << "x" << kernel_size << "x" << in_channels << "x" << out_channels << std::endl;
    std::cout << "批量大小: " << bsz << std::endl;

    std::vector<float> h_input(in_tensor_size);
    std::vector<float> h_output_naive(out_tensor_size);
    std::vector<float> h_output_cudnn(out_tensor_size);
    std::vector<float> h_kernel(kernel_elements);

    h_input = {
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
    };
    h_kernel = {
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    };

    float *d_input, *d_kernel, *d_output_cudnn, *d_output_naive;
    CHECK_CUDA(cudaMalloc(&d_input, in_tensor_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kernel, kernel_elements * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_cudnn, out_tensor_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_naive, out_tensor_size * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), in_tensor_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel.data(), kernel_elements * sizeof(float), cudaMemcpyHostToDevice));

    // 创建一系列描述符与句柄
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t kernelDesc;
    cudnnConvolutionDescriptor_t convDesc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&kernelDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, bsz, in_channels, height, width));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, bsz, out_channels, height, width));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(kernelDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channels, in_channels, kernel_size, kernel_size)); //
    // padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc, kernel_size / 2, kernel_size / 2, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT)); //

    // 测试最快的 cuDNN algo
    int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];

    // 让 cuDNN 测试多个前向卷积算法的运行时间和资源使用情况
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn, inputDesc, kernelDesc, convDesc, outputDesc,
                                                       requestedAlgoCount, &returnedAlgoCount, perfResults));

    cudnnConvolutionFwdAlgo_t algo = perfResults[0].algo;
    for (int i = 1; i < returnedAlgoCount; i++)
    {
        std::cout << "Algorithm: " << perfResults[i].algo << " Time: " << perfResults[i].time << std::endl;
        if (perfResults[i].status == CUDNN_STATUS_SUCCESS && perfResults[i].time < perfResults[0].time)
        {
            algo = perfResults[i].algo;
        }
    }

    std::cout << "选择的cuDNN算法为: " << algo << std::endl;

    // 卷积配置和选择的 algo 算法，在运行过程中最多需要多少工作空间（以字节为单位）
    size_t workspaceSize;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, kernelDesc, convDesc, outputDesc, algo, &workspaceSize)); //

    // 一块临时用的内存 无需暴露成 float*、int* 等
    void *d_workspace;
    CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSize));

    // 预热
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y,
                  out_channels * bsz);

    float alpha = 1.0f, beta = 0.0f;
    for (int i = 0; i < WARMUP_EPOCH; i++)
    {
        CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, kernelDesc, d_kernel, convDesc,
                                            algo, d_workspace, workspaceSize, &beta, outputDesc, d_output_cudnn));
        naiveConv2d<<<gridSize, blockSize>>>(d_input, d_kernel, d_output_naive, width, height, in_channels, out_channels, kernel_size, bsz);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // cuDNN 朴素卷积 性能测试
    float totalTime_cudnn = 0.0f;
    float totalTime_naive = 0.0f;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < BENCHMARK_EPOCH; i++)
    {
        // cuDNN benchmark
        CHECK_CUDA(cudaEventRecord(start));

        // alpha = 1.0f，beta = 0.0f 意味着：out = alpha * conv(in, kernel) + beta * out 为 标准卷积
        CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, kernelDesc, d_kernel, convDesc,
                                            algo, d_workspace, workspaceSize, &beta, outputDesc, d_output_cudnn));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        totalTime_cudnn += milliseconds;

        // Naive kernel benchmark
        CHECK_CUDA(cudaEventRecord(start));
        naiveConv2d<<<gridSize, blockSize>>>(d_input, d_kernel, d_output_naive, width, height, in_channels, out_channels, kernel_size, bsz);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        totalTime_naive += milliseconds;
    }

    // 平均时间
    float avgTime_cudnn = totalTime_cudnn / BENCHMARK_EPOCH;
    float avgTime_naive = totalTime_naive / BENCHMARK_EPOCH;

    printf("朴素算子conv2d %.3f ms 完成一次测试\n", avgTime_naive);
    printf("cuDNN conv2d %.3f ms 完成一次测试\n", avgTime_cudnn);

    CHECK_CUDA(cudaMemcpy(h_output_cudnn.data(), d_output_cudnn, out_tensor_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output_naive.data(), d_output_naive, out_tensor_size * sizeof(float), cudaMemcpyDeviceToHost));

    // 准确性测试
    float maxDiff = 0.0f;
    for (int i = 0; i < out_tensor_size; i++)
    {
        float diff = fabs(h_output_cudnn[i] - h_output_naive[i]);
        if (diff > maxDiff)
        {
            maxDiff = diff;
        }
    }

    printf("二者最大误差为: %e\n", maxDiff);

    printf("\ncuDNN 结果:\n");
    for (int b = 0; b < bsz; b++)
    {
        for (int c = 0; c < out_channels; c++)
        {
            printf("第 %d 个通道\n", c);
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int idx = ((b * out_channels + c) * height + h) * width + w;
                    printf("%f ", h_output_cudnn[idx]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    printf("\n朴素核函数结果:\n");
    for (int b = 0; b < bsz; b++)
    {
        for (int c = 0; c < out_channels; c++)
        {
            printf("第 %d 个通道\n", c);
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int idx = ((b * out_channels + c) * height + h) * width + w;
                    printf("%f ", h_output_naive[idx]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    printf("\n展平 cuDNN 的结果:\n");
    for (int i = 0; i < out_tensor_size; i++)
    {
        printf("%f", h_output_cudnn[i]);
        if (i < out_tensor_size - 1)
            printf(", ");
    }
    printf("\n");

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(inputDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(outputDesc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(kernelDesc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    CHECK_CUDNN(cudnnDestroy(cudnn));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_kernel));
    CHECK_CUDA(cudaFree(d_output_cudnn));
    CHECK_CUDA(cudaFree(d_output_naive));
    CHECK_CUDA(cudaFree(d_workspace));

    return 0;
}