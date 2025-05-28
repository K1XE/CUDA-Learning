import numpy as np
from torchvision import datasets, transforms

# ---------------------- 配置参数 ----------------------
BATCH_SIZE = 4          # 每个 batch 中样本数量
EPOCHS = 3              # 总训练轮数 (epoch)
LEARNING_RATE = 1e-3    # 学习率，控制参数更新步长
DATA_DIR = './data' # MNIST 数据保存路径
NP_SEED = 42            # NumPy 随机种子，保证可复现

# 固定随机种子
np.random.seed(NP_SEED)

# ---------------------- 数据加载与预处理 ----------------------
def load_mnist_data():
    """
    下载 MNIST 数据集，并归一化到 [0,1]，返回训练和测试集数组。
    - X_train 形状: (60000, 1, 28, 28), y_train: (60000,)
    - 此处只取前 10000 条用于快速调试。
    """
    # 定义数据转换：ToTensor 会返回 [0,1] float Tensor，但我们直接使用 .data.numpy()
    transform = transforms.Compose([transforms.ToTensor()])

    # 下载并加载数据集
    mnist_train = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

    # 转为 NumPy 并归一化到 [0,1]
    X_train = mnist_train.data.numpy().astype(np.float32) / 255.0
    y_train = mnist_train.targets.numpy().astype(np.int64)
    X_test = mnist_test.data.numpy().astype(np.float32) / 255.0
    y_test = mnist_test.targets.numpy().astype(np.int64)

    # 调整形状为 (N, 1, 28, 28)
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)

    # 为了快速实验，只取前 10000 条
    X_train, y_train = X_train[:10000], y_train[:10000]

    # 打印数据维度信息，便于调试
    print(f"训练集: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"测试集:  X_test={X_test.shape},  y_test={y_test.shape}\n")
    return X_train, y_train, X_test, y_test

# ---------------------- 激活函数及其导数 ----------------------
def relu(x):
    """ReLU 激活: max(0, x)"""
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLU 导数: x>0 时为 1，否则为 0"""
    return (x > 0).astype(np.float32)

# ---------------------- 全连接层操作 ----------------------
def initialize_weights(in_dim, out_dim):
    """Xavier/He 初始化，返回形状 (in_dim, out_dim) 的权重矩阵"""
    return np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)

def initialize_bias(dim):
    """初始化偏置为 0 向量，形状 (1, dim)"""
    return np.zeros((1, dim), dtype=np.float32)

def linear_forward(x, W, b):
    """前向线性变换: x @ W + b"""
    return x.dot(W) + b  # x: (batch, in_dim)

def linear_backward(grad_out, x, W):
    """
    线性层反向传播：
    - grad_out: 后续梯度，形状 (batch, out_dim)
    - x: 前向输入，形状 (batch, in_dim)
    - W: 参数权重，形状 (in_dim, out_dim)
    返回: grad_x, grad_W, grad_b
    """
    grad_W = x.T.dot(grad_out)                   # (in_dim, batch) @ (batch, out_dim) -> (in_dim, out_dim)
    grad_b = np.sum(grad_out, axis=0, keepdims=True)  # (1, out_dim)
    grad_x = grad_out.dot(W.T)                   # 传播到输入部分，形状 (batch, in_dim)
    return grad_x, grad_W, grad_b

# ---------------------- Softmax + 交叉熵 ----------------------
def softmax(x):
    """按行计算 softmax"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(logits, labels):
    """计算交叉熵损失，返回平均 loss"""
    batch_size = logits.shape[0]
    probs = softmax(logits)
    # 选出正确类别的概率并取负对数
    correct_probs = probs[np.arange(batch_size), labels]
    loss = -np.mean(np.log(correct_probs + 1e-9))  # 加上 epsilon 防止 log(0)
    return loss

# ---------------------- MLP 网络类 ----------------------
class MLP_Numpy:
    """简单两层 MLP，实现前向+反向+参数更新"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        # 初始化第一层参数
        self.W1 = initialize_weights(input_dim, hidden_dim)
        self.b1 = initialize_bias(hidden_dim)
        # 初始化第二层参数
        self.W2 = initialize_weights(hidden_dim, output_dim)
        self.b2 = initialize_bias(output_dim)

    def forward(self, x):
        """
        前向传播：
        1) 展平输入
        2) 线性变换 + ReLU
        3) 输出层线性
        返回 logits 和缓存用于反向
        """
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)         # (batch, 784)
        z1 = linear_forward(x_flat, self.W1, self.b1)  # (batch, hidden_dim)
        a1 = relu(z1)                              # ReLU 激活
        logits = linear_forward(a1, self.W2, self.b2)  # (batch, output_dim)
        cache = (x_flat, z1, a1)
        return logits, cache

    def backward(self, logits, labels, cache):
        """
        反向传播：
        1) 计算输出层误差
        2) 反向到第一层，并计算每层梯度
        返回各参数梯度
        """
        x_flat, z1, a1 = cache
        batch_size = logits.shape[0]

        # ---------- 输出层误差 ----------
        probs = softmax(logits)                    # softmax 概率
        # one-hot 标签矩阵
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(batch_size), labels] = 1
        # grad_logits: (batch, output_dim)
        grad_logits = (probs - one_hot) / batch_size

        # 输出层反向
        grad_a1, grad_W2, grad_b2 = linear_backward(grad_logits, a1, self.W2)

        # 隐藏层 ReLU 反向
        grad_z1 = grad_a1 * relu_derivative(z1)    # (batch, hidden_dim)

        # 第一层反向
        grad_x, grad_W1, grad_b1 = linear_backward(grad_z1, x_flat, self.W1)

        return grad_W1, grad_b1, grad_W2, grad_b2

    def update_params(self, grads, lr):
        """
        参数更新：梯度下降
        grads: (dW1, db1, dW2, db2)
        """
        dW1, db1, dW2, db2 = grads
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

# ---------------------- 训练与评估 ----------------------
def train_and_evaluate():
    # 加载数据
    X_train, y_train, X_test, y_test = load_mnist_data()
    input_dim = 28 * 28
    hidden_dim = 256
    output_dim = 10

    # 初始化模型
    model = MLP_Numpy(input_dim, hidden_dim, output_dim)
    print(f"模型参数维度: W1={model.W1.shape}, b1={model.b1.shape}, W2={model.W2.shape}, b2={model.b2.shape}\n")

    # 训练循环
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        num_batches = 0
        print(f"Epoch {epoch}/{EPOCHS}")

        # 按 batch 训练
        for start in range(0, X_train.shape[0], BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_X = X_train[start:end]
            batch_y = y_train[start:end]

            # 前向
            logits, cache = model.forward(batch_X)
            loss = cross_entropy_loss(logits, batch_y)
            epoch_loss += loss
            num_batches += 1

            # 反向
            grads = model.backward(logits, batch_y, cache)
            # 更新参数
            model.update_params(grads, LEARNING_RATE)

            # 每 100 个 batch 打印一次当前 loss
            if num_batches == 1 or num_batches % 100 == 0:
                print(f"  Batch {num_batches}  Loss: {loss:.4f}")

        # 打印本 epoch 平均 loss
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch} 平均 Loss: {avg_loss:.4f}")

        # 在测试集上评估
        test_logits, _ = model.forward(X_test)
        test_loss = cross_entropy_loss(test_logits, y_test)
        test_preds = np.argmax(test_logits, axis=1)
        test_acc = np.mean(test_preds == y_test)
        print(f"测试 Loss: {test_loss:.4f}, 测试准确率: {test_acc * 100:.2f}%\n")

    print("训练完成！")

if __name__ == '__main__':
    train_and_evaluate()
