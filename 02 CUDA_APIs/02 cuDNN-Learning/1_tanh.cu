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

__global__ void TanhNaiveKernel(float *input, float *output, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        output[i] = tanhf(input[i]);
    }
}

float cpuTanh(float x)
{
    return tanhf(x);
}

void init_data(float *data, int size)
{
    for (int i = 0; i < size; i++)
    {
        data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

bool VerifyRes(float *cpuData, float *gpuData, int size, float torlerance = 1e-5f)
{
    for (int i = 0; i < size; i++)
    {
        if (fabs(cpuData[i] - gpuData[i]) > torlerance)
        {
            printf("在%d索引处超出误差容忍度: CPU计算结果 = %f, GPU计算结果 = %f\n", i, cpuData[i], gpuData[i]);
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv)
{
    int bsz = 256;
    int channels = 32;
    int height = 128;
    int width = 128;
    int tensor_size = bsz * channels * height * width;

    std::vector<float> h_input(tensor_size);
    std::vector<float> h_output_naive(tensor_size);
    std::vector<float> h_output_cudnn(tensor_size);
    std::vector<float> h_output_cpu(tensor_size);

    init_data(h_input.data(), tensor_size);

    // cpu
    for (int i = 0; i < tensor_size; i ++ )
    {
        h_output_cpu[i] = cpuTanh(h_input[i]);
    }

    float *d_input, *d_output_naive, *d_output_cudnn;
    CHECK_CUDA(cudaMalloc(&d_input, tensor_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_cudnn, tensor_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_naive, tensor_size * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), tensor_size * sizeof(float), cudaMemcpyHostToDevice));

    // 创建事件 用于计时
    cudaEvent_t start, end;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));

    // 容器存储每次执行时间
    std::vector<float> naive_times(BENCHMARK_EPOCH, 0);
    std::vector<float> cudnn_times(BENCHMARK_EPOCH, 0);

    // 朴素Tanh
    dim3 block(256);
    dim3 grid(
        (tensor_size + block.x - 1) / block.x);

    for (int i = 0; i < WARMUP_EPOCH; i++)
    {
        TanhNaiveKernel<<<grid, block>>>(d_input, d_output_naive, tensor_size);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    for (int i = 0; i < BENCHMARK_EPOCH; i++)
    {
        CHECK_CUDA(cudaEventRecord(start));
        TanhNaiveKernel<<<grid, block>>>(d_input, d_output_naive, tensor_size);
        CHECK_CUDA(cudaEventRecord(end));
        CHECK_CUDA(cudaEventSynchronize(end));
        CHECK_CUDA(cudaEventElapsedTime(&naive_times[i], start, end));
    }

    // cuDNN
    cudnnHandle_t cudnn_handle1;
    CHECK_CUDNN(cudnnCreate(&cudnn_handle1));

    cudnnTensorDescriptor_t input_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc,
                                           CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           bsz, channels, height, width));

    cudnnActivationDescriptor_t activation_desc;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&activation_desc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(activation_desc,
                                             CUDNN_ACTIVATION_TANH,
                                             CUDNN_PROPAGATE_NAN, 0.0));

    float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < WARMUP_EPOCH; i++)
    {
        CHECK_CUDNN(cudnnActivationForward(cudnn_handle1, activation_desc,
                                           &alpha, input_desc, d_input,
                                           &beta, input_desc, d_output_cudnn));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    for (int i = 0; i < BENCHMARK_EPOCH; i++)
    {
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDNN(cudnnActivationForward(cudnn_handle1, activation_desc,
                                           &alpha, input_desc, d_input,
                                           &beta, input_desc, d_output_cudnn));
        CHECK_CUDA(cudaEventRecord(end));
        CHECK_CUDA(cudaEventSynchronize(end));
        CHECK_CUDA(cudaEventElapsedTime(&cudnn_times[i], start, end));
    }
    float naive_avg_time = 0.0f, cudnn_avg_time = 0.0f;
    for (int i = 0; i < BENCHMARK_EPOCH; i ++ )
    {
        naive_avg_time += naive_times[i];
        cudnn_avg_time += cudnn_times[i];
    }
    naive_avg_time /= BENCHMARK_EPOCH;
    cudnn_avg_time /= BENCHMARK_EPOCH;

    CHECK_CUDA(cudaMemcpy(h_output_naive.data(), d_output_naive, tensor_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output_cudnn.data(), d_output_cudnn, tensor_size * sizeof(float), cudaMemcpyDeviceToHost));

    // 验证准确性
    bool correct_naive = VerifyRes(h_output_cpu.data(), h_output_naive.data(), tensor_size);
    bool correct_cudnn = VerifyRes(h_output_cpu.data(), h_output_cudnn.data(), tensor_size);

    printf("张量大小： %d x %d x %d x %d\n", bsz, channels, height, width);
    printf("朴素算子tanh %.3f ms 完成一次测试\n", naive_avg_time);
    printf("cuDNN tanh %.3f ms 完成一次测试\n", cudnn_avg_time);
    printf("Speedup: %.2fx\n", naive_avg_time / cudnn_avg_time);
    printf("朴素算子tanh结果是： %s\n", correct_naive ? "正确的" : "错误的");
    printf("cuDNN tanh结果是： %s\n", correct_cudnn ? "正确的" : "错误的");


    CHECK_CUDNN(cudnnDestroy(cudnn_handle1));
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(activation_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output_cudnn));
    CHECK_CUDA(cudaFree(d_output_naive));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(end));

    return 0;
}


