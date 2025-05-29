#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 64
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 10000
#define TEST_SIZE 1000
#define BATCH_SIZE 4
#define EPOCHS 5
#define LEARNING_RATE 0.001

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

void load_labels(const char *filename, int *labels, int size)
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

// 返回一个服从 N(0,1) 的浮点随机数
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
float normal(float mean, float stddev)
{
    // u1, u2 均匀分布于 (0,1]
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);

    // Box-Muller 变换
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * M_PI * u2;
    float z0 = r * cosf(theta);
    // float z1 = r * sinf(theta); // 如需第二个样本可返回 z1

    // 缩放并平移到 mean, stddev
    return z0 * stddev + mean;
}

// 恺明初始化
void initialize_weights(float *weights, int size)
{
    float scaled = sqrtf(2.0f / size);
    for (int i = 0; i < size; i++)
    {
        weights[i] = normal(0, sqrtf(2.0f / size)); ///
    }
}

// 初始化偏置
void initialize_bias(float *bias, int size)
{
    for (int i = 0; i < size; i++)
    {
        bias[i] = 0.0f;
    }
}

// safe softmax
void safeSoftmax(float *mat, int bsz, int size)
{
    for (int b = 0; b < bsz; b++)
    {
        float _max = mat[b * size];
        for (int i = 1; i < size; i++)
        {
            if (mat[b * size + i] > _max)
            {
                _max = mat[b * size + i];
            }
        }
        float _sum = 0.0f;
        for (int i = 0; i < size; i++)
        {
            mat[b * size + i] = expf(mat[b * size + i] - _max);
            _sum += mat[b * size + i];
        }
        for (int i = 0; i < size; i++)
        {
            mat[b * size + i] = fmaxf(mat[b * size + i] / _sum, 1e-7f);
        }
    }
}

// A @ B
void matmul_A_B(float *A, float *B, float *C,
                const int M, const int N, const int K)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float _sums = 0.0f;
            for (int l = 0; l < K; l++)
            {
                _sums += A[i * K + l] * B[l * N + j];
            }
            C[i * N + j] = _sums;
        }
    }
}

// A @ B.T
void matmul_A_BT(float *A, float *B, float *C,
                 const int M, const int N, const int K)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float _sums = 0.0f;
            for (int l = 0; l < K; l++)
            {
                _sums += A[i * K + l] * B[j * K + l];
            }
            C[i * N + j] = _sums;
        }
    }
}

// A.T @ B
void matmul_AT_B(float *A, float *B, float *C,
                 const int M, const int N, const int K)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float _sums = 0.0f;
            for (int l = 0; l < K; l++)
            {
                _sums += A[l * M + i] * B[l * N + j];
            }
            C[i * N + j] = _sums;
        }
    }
}

// ReLU Activation Func
void relu_forward(float *mat, int size)
{
    for (int i = 0; i < size; i++)
    {
        mat[i] = fmaxf(0.0f, mat[i]);
    }
}

// Bias
void bias_forward(float *mat, float *bias, int bsz, int size)
{
    for (int b = 0; b < bsz; b++)
    {
        for (int i = 0; i < size; i++)
        {
            mat[b * size + i] += bias[i];
        }
    }
}

// nn forward
void forward(
    simpleNeuralNetwork *nn,
    float *input,
    float *hidden,
    float *output,
    int bsz)
{
    // X @ W1 + b1
    matmul_A_B(input, nn->weights1, hidden, bsz, HIDDEN_SIZE, INPUT_SIZE);
    bias_forward(hidden, nn->bias1, bsz, HIDDEN_SIZE);

    // 引入非线性
    relu_forward(hidden, bsz * HIDDEN_SIZE);

    // Hidden @ W2 + b2
    matmul_A_B(hidden, nn->weights2, output, bsz, OUTPUT_SIZE, HIDDEN_SIZE);
    bias_forward(output, nn->bias2, bsz, OUTPUT_SIZE);

    // 求概率
    safeSoftmax(output, bsz, OUTPUT_SIZE);
}

// 定义交叉熵为LOSS
float cross_entropy_loss(float *output, int *labels, int bsz)
{
    float total_loss = 0.0f;
    for (int b = 0; b < bsz; b++)
    {
        total_loss -= logf(fmaxf(output[b * OUTPUT_SIZE + labels[b]], 1e-7f));
    }
    return total_loss / bsz;
}

// 清空累计梯度
void zero_grad(float *grad, int size)
{
    memset(grad, 0, size * sizeof(float));
}

// ReLU 反向传播
void relu_backward(float *grad, float *mat, int size)
{
    for (int i = 0; i < size; i++)
    {
        grad[i] *= (mat[i] > 0);
    }
}

// Bias 反向传播
void bias_backward(float *grad_bias, float *grad, int bsz, int size)
{
    for (int i = 0; i < size; i++)
    {
        grad_bias[i] = 0.0f;
        for (int b = 0; b < bsz; b++)
        {
            grad_bias[i] += grad[b * size + i]; ///
        }
    }
}

// 计算输出层梯度
void compute_output_gradients(float *grad_output, float *output, int *labels, int bsz)
{
    for (int b = 0; b < bsz; b++)
    {
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            grad_output[b * OUTPUT_SIZE + i] = output[b * OUTPUT_SIZE + i];
        }
        grad_output[b * OUTPUT_SIZE + labels[b]] -= 1.0f; ///
    }
}

// 更新W和b的梯度
void update_gradients(
    float *grad_weights, // 指向存放权重梯度的数组，大小 = curr_size × prev_size
    float *grad_bias,    // 指向存放偏置梯度的数组，大小 = curr_size
    float *grad_layer,   // 当前层对损失的梯度 (δ)，大小 = batch_size × curr_size
    float *prev_layer,   // 前一层的激活值 (x)，大小 = batch_size × prev_size
    int bsz,            // 小批量样本数
    int prev_size,      // 前一层神经元个数（输入维度）
    int curr_size       // 当前层神经元个数（输出维度）
)
{
    for (int i = 0; i < curr_size; i++)
    { // （1）遍历当前层每个输出神经元 i
        for (int j = 0; j < prev_size; j++)
        { // （2）遍历前一层每个输入神经元 j
            for (int b = 0; b < bsz; b++)
            { // （3）遍历每个样本 b
                // 累加：∂L/∂W[i][j] += δ_i(b) * x_j(b)
                grad_weights[i * prev_size + j] += grad_layer[b * curr_size + i] * prev_layer[b * prev_size + j];
            }
        }
        for (int b = 0; b < bsz; b++)
        { // （4）对偏置的梯度累加
            // ∂L/∂bias[i] += δ_i(b)
            grad_bias[i] += grad_layer[b * curr_size + i];
        }
    }
}

// 反向传播
//   nn: 指向神经网络结构体，包含权重、偏置及对应的梯度缓冲
//   input: 输入层激活值，形状 (batch_size, INPUT_SIZE)
//   hidden: 隐藏层激活值（ReLU 作用后），形状 (batch_size, HIDDEN_SIZE)
//   output: 输出层激活值（Softmax 作用后），形状 (batch_size, OUTPUT_SIZE)
//   labels: 目标标签，长度 batch_size
//   bsz: 当前小批量的样本数量
void backward(simpleNeuralNetwork *nn,
              float *input,  // (batch_size, INPUT_SIZE)
              float *hidden, // (batch_size, HIDDEN_SIZE)
              float *output, // (batch_size, OUTPUT_SIZE)
              int *labels,  // (batch_size,)
              int bsz)      ///
{
    // 1. 清零所有梯度缓存，确保累加的是当前批次的梯度
    zero_grad(nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE);
    zero_grad(nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE);
    zero_grad(nn->grad_bias1, HIDDEN_SIZE);
    zero_grad(nn->grad_bias2, OUTPUT_SIZE);

    // 2. 分配输出层误差缓冲：形状 (batch_size, OUTPUT_SIZE)
    float *grad_output = (float *)malloc(bsz * OUTPUT_SIZE * sizeof(float));
    //    compute_output_gradients 内部计算每个样本的 ∂L/∂Z2 = y_pred - y_true（交叉熵 + softmax 的简化形式）
    compute_output_gradients(grad_output, output, labels, bsz);

    // 3. 计算并累加输出层权重梯度：W2.grad += H^T @ δ2
    //    hidden: (batch_size, HIDDEN_SIZE)
    //    grad_output: (batch_size, OUTPUT_SIZE)
    //    结果写入 nn->grad_weights2，形状 (HIDDEN_SIZE, OUTPUT_SIZE)
    matmul_AT_B(hidden, grad_output,
                nn->grad_weights2,
                HIDDEN_SIZE, OUTPUT_SIZE, bsz);

    // 4. 计算并累加输出层偏置梯度：b2.grad[i] += sum_b δ2[b,i]
    bias_backward(nn->grad_bias2, grad_output,
                  bsz, OUTPUT_SIZE);

    // 5. 计算隐藏层误差 δ1 = δ2 @ W2^T，结果形状 (batch_size, HIDDEN_SIZE)
    float *dX2 = (float *)malloc(bsz * HIDDEN_SIZE * sizeof(float));
    matmul_A_BT(grad_output, nn->weights2,
                dX2,
                bsz, HIDDEN_SIZE, OUTPUT_SIZE);

    // 6. 应用 ReLU 导数：δ1 ∘ ReLU'(Z1)
    //    ReLU'(z) = 1 if z > 0, else 0
    float *d_ReLU_out = (float *)malloc(bsz * HIDDEN_SIZE * sizeof(float));
    for (int idx = 0; idx < bsz * HIDDEN_SIZE; idx++)
    {
        // hidden[idx] 即 ReLU(Z1)[idx]
        d_ReLU_out[idx] = dX2[idx] * (hidden[idx] > 0 ? 1.0f : 0.0f);
    }

    // 7. 计算并累加第一层权重梯度：W1.grad += X^T @ δ1
    //    input: (batch_size, INPUT_SIZE)
    //    d_ReLU_out: (batch_size, HIDDEN_SIZE)
    //    结果写入 nn->grad_weights1，形状 (INPUT_SIZE, HIDDEN_SIZE)
    matmul_AT_B(input, d_ReLU_out,
                nn->grad_weights1,
                INPUT_SIZE, HIDDEN_SIZE, bsz);

    // 8. 计算并累加第一层偏置梯度：b1.grad[j] += sum_b δ1[b,j]
    bias_backward(nn->grad_bias1, d_ReLU_out,
                  bsz, HIDDEN_SIZE);

    // 9. 释放临时缓冲区，防止内存泄漏
    free(grad_output);
    free(dX2);
    free(d_ReLU_out);
}

// 梯度下降更新参数
void update_para(simpleNeuralNetwork *nn)
{
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++)
    {
        nn->weights1[i] -= LEARNING_RATE * nn->grad_weights1[i];
    }
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++)
    {
        nn->weights2[i] -= LEARNING_RATE * nn->grad_weights2[i];
    }
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        nn->bias1[i] -= LEARNING_RATE * nn->grad_bias1[i];
    }
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        nn->bias2[i] -= LEARNING_RATE * nn->grad_bias2[i];
    }
}

// train
//   nn: 指向神经网络结构体，包含权重、偏置及对应的梯度缓冲
//   X_train: 训练集特征数组，形状 (TRAIN_SIZE, INPUT_SIZE)
//   y_train: 训练集标签数组，长度 TRAIN_SIZE
void train(simpleNeuralNetwork *nn, float *X_train, int *y_train)
{
    // 1. 为前向传播中间结果分配缓冲
    // hidden 保存隐藏层激活值 (batch_size, HIDDEN_SIZE)
    float *hidden = (float *)malloc(BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    // output 保存输出层激活值 (batch_size, OUTPUT_SIZE)
    float *output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    // 2. 计算总批次数，假设 TRAIN_SIZE 能被 BATCH_SIZE 整除
    int num_batches = TRAIN_SIZE / BATCH_SIZE;

    // 3. 遍历每个 epoch
    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        float total_loss = 0.0f; // 累加本 epoch 的总损失
        int correct = 0;         // 累加预测正确的样本数

        // 3.1 遍历所有 batch
        for (int batch = 0; batch < num_batches; batch++)
        {
            int start_idx = batch * BATCH_SIZE;

            // 3.1.1 前向传播：计算当前 batch 的隐藏层和输出层结果
            // 输入 X_batch 形状 (BATCH_SIZE, INPUT_SIZE)
            forward(nn,
                    &X_train[start_idx * INPUT_SIZE], // 输入数据指针
                    hidden,                           // 隐藏层输出缓冲
                    output,                           // 输出层输出缓冲
                    BATCH_SIZE);

            // 3.1.2 损失计算：交叉熵损失
            float loss = cross_entropy_loss(
                output,
                &y_train[start_idx],
                BATCH_SIZE);
            total_loss += loss;

            // 3.1.3 计算分类准确率：逐样本选取最大概率的类别
            for (int i = 0; i < BATCH_SIZE; i++)
            {
                // 寻找 output[i] 中值最大的索引
                int predicted = 0;
                for (int j = 1; j < OUTPUT_SIZE; j++)
                {
                    if (output[i * OUTPUT_SIZE + j] > output[i * OUTPUT_SIZE + predicted])
                    {
                        predicted = j;
                    }
                }
                // 与真实标签比较
                if (predicted == y_train[start_idx + i])
                {
                    correct++;
                }
            }

            // 3.1.4 后向传播：基于本 batch 计算梯度并累加到 nn->grad_*
            backward(nn,
                     &X_train[start_idx * INPUT_SIZE],
                     hidden,
                     output,
                     &y_train[start_idx],
                     BATCH_SIZE);

            // 3.1.5 更新参数：权重和偏置根据累加的梯度进行一次梯度下降
            update_para(nn);

            // 3.1.6 进度输出：每 100 个 batch 或者第 1 个 batch 打印一次
            if ((batch + 1) % 100 == 0 || (epoch == 0 && batch == 0))
            {
                // 平均损失 = total_loss / (batch+1)
                // 累计准确率 = correct / ((batch+1) * BATCH_SIZE)
                printf("Epoch %2d/%2d | Batch %4d/%4d | Loss: %.4f | Acc: %6.2f%%\r",
                       epoch + 1, EPOCHS,
                       batch + 1, num_batches,
                       total_loss / (batch + 1),
                       100.0f * correct / ((batch + 1) * BATCH_SIZE));
                fflush(stdout);
            }
        }

        // 3.2 当前 epoch 完成后的汇总输出
        printf("\n--- Epoch %2d/%2d Complete: Avg Loss = %.4f, Acc = %.2f%% ---\n",
               epoch + 1, EPOCHS,
               total_loss / num_batches,
               100.0f * correct / TRAIN_SIZE);
    }

    // 4. 释放中间缓冲，避免内存泄漏
    free(hidden);
    free(output);
}

// 初始化矩阵
void initialize_neural_network(simpleNeuralNetwork *nn)
{
    nn->weights1 = (float *)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    nn->weights2 = (float *)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    nn->bias1 = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    nn->bias2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    nn->grad_weights1 = (float *)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    nn->grad_weights2 = (float *)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    nn->grad_bias1 = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    nn->grad_bias2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));

    initialize_weights(nn->weights1, HIDDEN_SIZE * INPUT_SIZE);
    initialize_weights(nn->weights2, OUTPUT_SIZE * HIDDEN_SIZE);
    initialize_bias(nn->bias1, HIDDEN_SIZE);
    initialize_bias(nn->bias2, OUTPUT_SIZE);
}

int main(int argc, char **argv)
{
    srand(time(NULL));

    simpleNeuralNetwork nn;
    initialize_neural_network(&nn);

    float *X_train = (float *)malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    int *y_train = (int *)malloc(TRAIN_SIZE * sizeof(int));
    float *X_test = (float *)malloc(TEST_SIZE * INPUT_SIZE * sizeof(float));
    int *y_test = (int *)malloc(TEST_SIZE * sizeof(int));

    load_data("./mnist_data/X_train.bin", X_train, TRAIN_SIZE * INPUT_SIZE);
    load_labels("./mnist_data/y_train.bin", y_train, TRAIN_SIZE);
    load_data("./mnist_data/X_test.bin", X_test, TEST_SIZE * INPUT_SIZE);
    load_labels("./mnist_data/y_test.bin", y_test, TEST_SIZE);

    // 打印一个数字
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

    train(&nn, X_train, y_train);

    free(nn.weights1);
    free(nn.weights2);
    free(nn.bias1);
    free(nn.bias2);
    free(nn.grad_weights1);
    free(nn.grad_weights2);
    free(nn.grad_bias1);
    free(nn.grad_bias2);
    free(X_train);
    free(y_train);
    free(X_test);
    free(y_test);

    return 0;
}