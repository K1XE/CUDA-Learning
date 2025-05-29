#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <vector>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 4096
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 10000
#define TEST_SIZE 1000
#define BATCH_SIZE 32
#define EPOCHS 20
#define LEARNING_RATE 0.001

#define CHECK_CUDA(call)                                                                 \
    do                                                                                   \
    {                                                                                    \
        cudaError_t err = call;                                                          \
        if (err != cudaSuccess)                                                          \
        {                                                                                \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl;                           \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while (0)

typedef struct
{
    float *weights1;
    float *weights2;
    float *bias1;
    float *bias2;
    float *grad_weights1;
    float *grad_weights2;
    float *grad_bias1;
    float *grad_bias2;
} simpleNeuralNetwork;

void load_data(const char *filename, float *data, int size)
{
    FILE *file = fopen(filename, "rb");
    if (file == NULL)
    {
        fprintf(stderr, "此文件不存在: %s\n", filename);
        exit(1);
    }

    size_t read_size = fread(data, sizeof(float), size, file);
    if (read_size != size)
    {
        fprintf(stderr, "读取数据异常: 应有 %d 元素, 实有 %zu 元素\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

void load_labels(const char *filename, uint *labels, int size)
{
    FILE *file = fopen(filename, "rb");
    if (file == NULL)
    {
        fprintf(stderr, "此文件不存在: %s\n", filename);
        exit(1);
    }

    size_t read_size = fread(labels, sizeof(int), size, file);
    if (read_size != size)
    {
        fprintf(stderr, "读取数据异常: 应有 %d 元素, 实有 %zu 元素\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

float normal(float mean, float stddev)
{
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * M_PI * u2;
    float z0 = r * cosf(theta);
    return z0 * stddev + mean;
}

// 初始化权重
void initialize_weights(float *weights, uint size)
{
    float scaled = sqrtf(2.0f / size);
    for (int i = 0; i < size; i++)
    {
        weights[i] = normal(0, sqrtf(2.0f / size));
    }
}

// 初始化偏置
void initialize_bias(float *bias, uint size)
{
    for (int i = 0; i < size; i++)
    {
        bias[i] = 0.0f;
    }
}

// Naive A @ B in CUDA
__global__ void matmulABKernel(float *A, float *B, float *C,
                               const uint M, const uint N, const uint K)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < M && col < N)
    {
        float _sums = 0.0f;
        for (int i = 0; i < K; i++)
        {
            _sums += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = _sums;
    }
}

// Naive A @ B.T in CUDA
__global__ void matmulABTKernel(float *A, float *B, float *C,
                                const uint M, const uint N, const uint K)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < M && col < N)
    {
        float _sums = 0.0f;
        for (int i = 0; i < K; i++)
        {
            _sums += A[row * K + i] * B[col * K + i];
        }
        C[row * N + col] = _sums;
    }
}

// Naive A.T @ B in CUDA
__global__ void matmulATBKernel(float *A, float *B, float *C,
                                const uint M, const uint N, const uint K)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < M && col < N)
    {
        float _sums = 0.0f;
        for (int i = 0; i < K; i++)
        {
            _sums += A[i * M + row] * B[i * N + col];
        }
        C[row * N + col] = _sums;
    }
}

// ReLU in CUDA
__global__ void reluKernel(float *x, const uint sz)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < sz)
        x[idx] = fmaxf(0.0f, x[idx]);
}

// bias in CUDA
__global__ void biasAddKernel(float *x, float *bias, const uint bsz, uint sz)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int b = idx / sz;
    int i = idx % sz;
    if (b < bsz && i < sz)
        x[idx] += bias[i];
}

// safe softmax in CUDA
__global__ void safeSoftMaxKernel(float *x, const uint bsz, const uint sz)
{
    int b = blockIdx.x;
    if (b < bsz)
    {
        float _max = x[b * sz];
        for (int i = 1; i < sz; i++)
            _max = fmaxf(_max, x[b * sz + i]);
        float _sums = 0.0f;
        for (int i = 0; i < sz; i++)
        {
            x[b * sz + i] = expf(x[b * sz + i] - _max);
            _sums += x[b * sz + i];
        }
        for (int i = 0; i < sz; i++)
            x[b * sz + i] = fmaxf(x[b * sz + i] / _sums, 1e-7f);
    }
}

// forward in CUDA
void forward(simpleNeuralNetwork *nn, float *d_input, float *d_hidden, float *d_output, const uint bsz)
{
    dim3 blocks(32, 32);
    dim3 grids((HIDDEN_SIZE + blocks.x - 1) / blocks.x, (bsz + blocks.y - 1) / blocks.y);

    // X @ W1 + b1
    matmulABKernel<<<grids, blocks>>>(d_input, nn->weights1, d_hidden, bsz, HIDDEN_SIZE, INPUT_SIZE);
    CHECK_CUDA(cudaGetLastError());

    biasAddKernel<<<(bsz * HIDDEN_SIZE + 255) / 256,
                    256>>>(d_hidden, nn->bias1, bsz, HIDDEN_SIZE);
    CHECK_CUDA(cudaGetLastError());

    // ReLU
    reluKernel<<<(bsz * HIDDEN_SIZE + 255) / 256,
                 256>>>(d_hidden, bsz * HIDDEN_SIZE);
    CHECK_CUDA(cudaGetLastError());

    // Hidden @ W2 + b2
    dim3 _blocks(32, 32);
    dim3 _grids((OUTPUT_SIZE + _blocks.x - 1) / _blocks.x,
                (bsz + _blocks.y - 1) / _blocks.y);
    matmulABKernel<<<_grids, _blocks>>>(d_hidden, nn->weights2, d_output, bsz, OUTPUT_SIZE, HIDDEN_SIZE);
    CHECK_CUDA(cudaGetLastError());

    biasAddKernel<<<(bsz * OUTPUT_SIZE + 255) / 256,
                    256>>>(d_output, nn->bias2, bsz, OUTPUT_SIZE);
    CHECK_CUDA(cudaGetLastError());

    // safe softmax
    safeSoftMaxKernel<<<bsz, OUTPUT_SIZE>>>(d_output, bsz, OUTPUT_SIZE);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaDeviceSynchronize());
}

// cross entropy loss
float cross_entropy_loss(float *output, uint *labels, const uint bsz)
{
    float _loss_sums = 0.0f;
    for (int b = 0; b < bsz; b++)
        _loss_sums -= logf(fmaxf(output[b * OUTPUT_SIZE + labels[b]], 1e-7));
    return _loss_sums / bsz;
}

// step the grad in CUDA
__global__ void zeroGradKernel(float *grad, const uint sz)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < sz)
        grad[idx] = 0.0f;
}

// compute output grad in CUDA
__global__ void computeOutputGradKernel(float *grad_output, float *output, uint *labels, const uint bsz)
{
    int b = threadIdx.x + blockIdx.x * blockDim.x;
    if (b < bsz)
    {
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            grad_output[b * OUTPUT_SIZE + i] = output[b * OUTPUT_SIZE + i];
        }
        grad_output[b * OUTPUT_SIZE + labels[b]] -= 1.0f;
    }
}

// update grad in CUDA
__global__ void updateGradKernel(
    float *grad_weights, // csz × psz
    float *grad_bias,    // csz
    float *grad_layer,   // bsz × csz
    float *prev_layer,   // bsz × psz
    const uint bsz, const uint psz, const uint csz)
{
    int i = blockIdx.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < csz && j < psz)
    {
        float _w_grad_sums = 0.0f;
        for (int b = 0; b < bsz; b++)
            _w_grad_sums += grad_layer[b * csz + i] * prev_layer[b * psz + j];
        atomicAdd(&grad_weights[i * psz + j], _w_grad_sums); ///
        if (j == 0)
        {
            float _b_grad_sums = 0.0f;
            for (int b = 0; b < bsz; b++)
                _b_grad_sums += grad_layer[b * csz + i];
            atomicAdd(&grad_bias[i], _b_grad_sums);
        }
    }
}

// ReLU grad in CUDA
__global__ void dReLUKernel(float *x, float *d_relu_out, const uint sz)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < sz)
        d_relu_out[idx] = (x[idx] > 0.0f ? 1.0f : 0.0f);
}

// d_dX2 * d_grad_hidden in CUDA
__global__ void elemWiseMulGradKernel(float *grad1, float *grad2, const uint sz) ///
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < sz)
        grad1[idx] *= grad2[idx];
}

// backward in CUDA
void backward(simpleNeuralNetwork *nn, float *d_input, float *d_hidden, float *d_output, uint *d_labels, const uint bsz)
{
    // step
    zeroGradKernel<<<(HIDDEN_SIZE * INPUT_SIZE + 256 - 1) / 256,
                     256>>>(nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE);
    CHECK_CUDA(cudaGetLastError());

    zeroGradKernel<<<(OUTPUT_SIZE * HIDDEN_SIZE + 256 - 1) / 256,
                     256>>>(nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE);
    CHECK_CUDA(cudaGetLastError());

    zeroGradKernel<<<(HIDDEN_SIZE + 256 - 1) / 256,
                     256>>>(nn->grad_bias1, HIDDEN_SIZE);
    CHECK_CUDA(cudaGetLastError());

    zeroGradKernel<<<(OUTPUT_SIZE + 256 - 1) / 256,
                     256>>>(nn->grad_bias2, OUTPUT_SIZE);
    CHECK_CUDA(cudaGetLastError());

    // output layer grad
    float *d_grad_output;
    CHECK_CUDA(cudaMalloc(&d_grad_output, bsz * OUTPUT_SIZE * sizeof(float)));
    computeOutputGradKernel<<<(bsz + 255) / 256,
                              256>>>(d_grad_output, d_output, d_labels, bsz);
    CHECK_CUDA(cudaGetLastError());

    // W2.grad = hidden.T @ output_grad
    dim3 blocks(32, 32);
    dim3 grids((HIDDEN_SIZE + blocks.x - 1) / blocks.x,
               (OUTPUT_SIZE + blocks.y - 1) / blocks.y);
    matmulATBKernel<<<grids, blocks>>>(d_hidden, d_grad_output, nn->grad_weights2, HIDDEN_SIZE, OUTPUT_SIZE, bsz);
    CHECK_CUDA(cudaGetLastError());

    // bias2 grad
    updateGradKernel<<<grids, blocks>>>(nn->grad_weights2, nn->grad_bias2, d_grad_output, d_hidden,
                                        bsz, HIDDEN_SIZE, OUTPUT_SIZE);
    CHECK_CUDA(cudaGetLastError());

    // dX2 grad δ1 = δ2 @ W2^T
    float *d_dX2;
    CHECK_CUDA(cudaMalloc(&d_dX2, bsz * HIDDEN_SIZE * sizeof(float)));
    dim3 _block(32, 32);
    dim3 _grid((HIDDEN_SIZE + _block.x - 1) / _block.x,
               (bsz + _block.y - 1) / _block.y);
    matmulABTKernel<<<_grid, _block>>>(d_grad_output, nn->weights2, d_dX2, bsz, HIDDEN_SIZE, OUTPUT_SIZE);
    CHECK_CUDA(cudaGetLastError());

    // d_relu_out grad
    float *d_grad_hidden;
    CHECK_CUDA(cudaMalloc(&d_grad_hidden, bsz * HIDDEN_SIZE * sizeof(float)));
    dReLUKernel<<<(bsz * HIDDEN_SIZE + 255) / 256,
                  256>>>(d_hidden, d_grad_hidden, bsz * HIDDEN_SIZE);
    CHECK_CUDA(cudaGetLastError());

    // gard mul
    elemWiseMulGradKernel<<<(bsz * HIDDEN_SIZE + 255) / 256,
                            256>>>(d_dX2, d_grad_hidden, bsz * HIDDEN_SIZE);
    CHECK_CUDA(cudaGetLastError());

    // W1.grad += X^T @ δ1
    dim3 __block(32, 32);
    dim3 __grid((INPUT_SIZE + __block.x - 1) / __block.x,
                (HIDDEN_SIZE + __block.y - 1) / __block.y);
    matmulATBKernel<<<__grid, __block>>>(d_input, d_dX2, nn->grad_weights1, INPUT_SIZE, HIDDEN_SIZE, bsz);
    CHECK_CUDA(cudaGetLastError());

    // bias1 grad
    updateGradKernel<<<__grid, __block>>>(nn->grad_weights1, nn->grad_bias1, d_dX2, d_input,
                                          bsz, INPUT_SIZE, HIDDEN_SIZE);
    CHECK_CUDA(cudaGetLastError());

    // free mem
    CHECK_CUDA(cudaFree(d_grad_output));
    CHECK_CUDA(cudaFree(d_dX2));
    CHECK_CUDA(cudaFree(d_grad_hidden));

    CHECK_CUDA(cudaDeviceSynchronize());
}

// gradient descent
__global__ void updataParaKernel(float *weights, float *grad_weights, const uint sz)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < sz)
        weights[idx] -= LEARNING_RATE * grad_weights[idx];
}

void update_all_para(simpleNeuralNetwork *nn)
{
    int _block = 256;
    int _grid;

    // W1
    _grid = (HIDDEN_SIZE * INPUT_SIZE + _block - 1) / _block;
    updataParaKernel<<<_grid, _block>>>(nn->weights1, nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE);
    CHECK_CUDA(cudaGetLastError());

    // b1
    _grid = (HIDDEN_SIZE + _block - 1) / _block;
    updataParaKernel<<<_grid, _block>>>(nn->bias1, nn->grad_bias1, HIDDEN_SIZE);
    CHECK_CUDA(cudaGetLastError());

    // W2
    _grid = (OUTPUT_SIZE * HIDDEN_SIZE + _block - 1) / _block;
    updataParaKernel<<<_grid, _block>>>(nn->weights2, nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE);
    CHECK_CUDA(cudaGetLastError());

    // b2
    _grid = (OUTPUT_SIZE + _block - 1) / _block;
    updataParaKernel<<<_grid, _block>>>(nn->bias2, nn->grad_bias2, OUTPUT_SIZE);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaDeviceSynchronize());
}

// evaluate accuracy
float eval_acc(simpleNeuralNetwork *nn, float *d_X_test, uint *d_y_test,
               float *d_hidden, float *d_output, const uint tsz)
{
    int num_bsz = (tsz + BATCH_SIZE - 1) / BATCH_SIZE;
    int total_correct = 0;
    int total_processed = 0;

    for (int b = 0; b < num_bsz; b++)
    {
        int cur_bsz = (b == num_bsz - 1) ? (tsz - b * BATCH_SIZE) : BATCH_SIZE; // clip
        if (cur_bsz <= 0)
            break;

        forward(nn, &d_X_test[b * BATCH_SIZE * INPUT_SIZE],
                d_hidden, d_output, cur_bsz);

        std::vector<float> h_output(cur_bsz * OUTPUT_SIZE);
        std::vector<uint> h_y_test(cur_bsz);

        CHECK_CUDA(cudaMemcpy(h_output.data(), d_output,
                              cur_bsz * OUTPUT_SIZE * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_y_test.data(), &d_y_test[b * BATCH_SIZE],
                              cur_bsz * sizeof(uint),
                              cudaMemcpyDeviceToHost));

        for (int i = 0; i < cur_bsz; i++)
        {
            int predicted = 0;
            for (int j = 1; j < OUTPUT_SIZE; j++)
            {
                if (h_output[i * OUTPUT_SIZE + j] > h_output[i * OUTPUT_SIZE + predicted])
                    predicted = j;
            }
            if (predicted == h_y_test[i])
                total_correct++;
        }

        total_processed += cur_bsz;
    }
    return 100.0f * total_correct / total_processed;
}

// train
void train(simpleNeuralNetwork *nn,
           float *X_train, uint *y_train, float *X_test, uint *y_test)
{
    float *d_X_train, *d_X_test, *d_hidden, *d_output;
    uint *d_y_train, *d_y_test;

    // Allocate memory
    CHECK_CUDA(cudaMalloc(&d_X_train, TRAIN_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_X_test, TEST_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y_train, TRAIN_SIZE * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_y_test, TEST_SIZE * sizeof(int)));

    // cpy data to GPU
    CHECK_CUDA(cudaMemcpy(d_X_train, X_train,
                          TRAIN_SIZE * INPUT_SIZE * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_X_test, X_test,
                          TEST_SIZE * INPUT_SIZE * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y_train, y_train,
                          TRAIN_SIZE * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y_test, y_test,
                          TEST_SIZE * sizeof(int),
                          cudaMemcpyHostToDevice));

    int num_batches = TRAIN_SIZE / BATCH_SIZE;

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        float total_loss = 0.0f;

        // zero out grad in each epoch
        zeroGradKernel<<<(HIDDEN_SIZE * INPUT_SIZE + 256 - 1) / 256,
                         256>>>(nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE);
        zeroGradKernel<<<(OUTPUT_SIZE * HIDDEN_SIZE + 256 - 1) / 256,
                         256>>>(nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE);
        zeroGradKernel<<<(HIDDEN_SIZE + 256 - 1) / 256, 256>>>(nn->grad_bias1, HIDDEN_SIZE);
        zeroGradKernel<<<(OUTPUT_SIZE + 256 - 1) / 256, 256>>>(nn->grad_bias2, OUTPUT_SIZE);
        CHECK_CUDA(cudaDeviceSynchronize());

        for (int b = 0; b < num_batches; b++)
        {
            int start_idx = b * BATCH_SIZE;

            forward(nn, &d_X_train[start_idx * INPUT_SIZE], d_hidden, d_output, BATCH_SIZE);

            std::vector<float> h_output(BATCH_SIZE * OUTPUT_SIZE);
            CHECK_CUDA(cudaMemcpy(h_output.data(), d_output,
                                  BATCH_SIZE * OUTPUT_SIZE * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            float loss = cross_entropy_loss(h_output.data(), &y_train[start_idx], BATCH_SIZE);
            total_loss += loss;

            backward(nn,
                     &d_X_train[start_idx * INPUT_SIZE],
                     d_hidden,
                     d_output,
                     &d_y_train[start_idx],
                     BATCH_SIZE);

            update_all_para(nn);

            if ((b + 1) % 100 == 0 || (epoch == 0 && b == 0))
            {
                int test_start_idx = rand() % (TEST_SIZE - BATCH_SIZE);
                float test_acc = eval_acc(nn,
                                          &d_X_test[test_start_idx * INPUT_SIZE],
                                          &d_y_test[test_start_idx],
                                          d_hidden,
                                          d_output,
                                          BATCH_SIZE);
                printf("Epoch %2d/%2d | Batch %4d/%4d | Loss: %.4f | Test Acc: %6.2f%%\r",
                       epoch + 1, EPOCHS,
                       b + 1, num_batches,
                       total_loss / (b + 1),
                       test_acc);
                fflush(stdout);
            }
        }

        // acc of test set in a epoch
        float test_acc = eval_acc(nn, d_X_test, d_y_test, d_hidden, d_output, TEST_SIZE);

        printf("\n--- Epoch %2d/%2d Complete: Avg Loss = %.4f, Test Acc = %.2f%% ---\n",
               epoch + 1, EPOCHS,
               total_loss / num_batches,
               test_acc);
    }

    // free GPU mem
    CHECK_CUDA(cudaFree(d_X_train));
    CHECK_CUDA(cudaFree(d_X_test));
    CHECK_CUDA(cudaFree(d_hidden));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_y_train));
    CHECK_CUDA(cudaFree(d_y_test));
}

void initialize_neural_network(simpleNeuralNetwork *nn)
{
    // device
    CHECK_CUDA(cudaMalloc(&nn->weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nn->weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nn->bias1, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nn->bias2, OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nn->grad_bias1, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&nn->grad_bias2, OUTPUT_SIZE * sizeof(float)));

    // host
    std::vector<float> h_weights1(HIDDEN_SIZE * INPUT_SIZE);
    std::vector<float> h_weights2(OUTPUT_SIZE * HIDDEN_SIZE);
    std::vector<float> h_bias1(HIDDEN_SIZE);
    std::vector<float> h_bias2(OUTPUT_SIZE);

    // init w and b
    initialize_weights(h_weights1.data(), HIDDEN_SIZE * INPUT_SIZE);
    initialize_weights(h_weights2.data(), OUTPUT_SIZE * HIDDEN_SIZE);
    initialize_bias(h_bias1.data(), HIDDEN_SIZE);
    initialize_bias(h_bias2.data(), OUTPUT_SIZE);

    // h2d
    CHECK_CUDA(cudaMemcpy(nn->weights1, h_weights1.data(), HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(nn->weights2, h_weights2.data(), OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(nn->bias1, h_bias1.data(), HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(nn->bias2, h_bias2.data(), OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
}

int main(int argc, char **argv)
{
    srand(time(NULL));

    // init nn
    simpleNeuralNetwork nn;
    initialize_neural_network(&nn);

    // mnist data
    std::vector<float> X_train(TRAIN_SIZE * INPUT_SIZE);
    std::vector<float> X_test(TEST_SIZE * INPUT_SIZE);
    std::vector<uint> y_train(TRAIN_SIZE);
    std::vector<uint> y_test(TEST_SIZE);

    // load data
    load_data("./mnist_data/X_train.bin", X_train.data(), TRAIN_SIZE * INPUT_SIZE);
    load_labels("./mnist_data/y_train.bin", y_train.data(), TRAIN_SIZE);
    load_data("./mnist_data/X_test.bin", X_test.data(), TEST_SIZE * INPUT_SIZE);
    load_labels("./mnist_data/y_test.bin", y_test.data(), TEST_SIZE);

    // print num
    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            if (X_train[0 * INPUT_SIZE + i * 28 + j] > 0.0f)
            {
                printf("X");
            }
            else
            {
                printf(" ");
            }
        }
        printf("\n");
    }

    printf("训练集前10个真实标签: ");
    for (int i = 0; i < 10; i++)
    {
        printf("%d ", y_train[i]);
    }
    printf("\n");

    // timer start
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    train(&nn, X_train.data(), y_train.data(), X_test.data(), y_test.data());

    // timer end
    clock_gettime(CLOCK_MONOTONIC, &end);

    // calc time
    double training_time = (end.tv_sec - start.tv_sec) +
                           (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("\n训练所需时间: %.2f s\n", training_time);

    // free GPU mem
    CHECK_CUDA(cudaFree(nn.weights1));
    CHECK_CUDA(cudaFree(nn.weights2));
    CHECK_CUDA(cudaFree(nn.bias1));
    CHECK_CUDA(cudaFree(nn.bias2));
    CHECK_CUDA(cudaFree(nn.grad_weights1));
    CHECK_CUDA(cudaFree(nn.grad_weights2));
    CHECK_CUDA(cudaFree(nn.grad_bias1));
    CHECK_CUDA(cudaFree(nn.grad_bias2));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}