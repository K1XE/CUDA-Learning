import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------- 全局配置 ----------------------
EPOCHS = 3               # 总共训练轮数 (epoch)，每轮遍历一次完整训练集
LEARNING_RATE = 1e-3      # 学习率，控制参数更新步长，过大容易发散，过小收敛慢
BATCH_SIZE = 4            # 批量大小，一次性输入模型的样本数，影响显存占用与梯度稳定性
DATA_DIR = './data'  # 数据集下载及保存路径
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动检测可用设备：GPU 优先

# ---------------------- 数据加载与预处理 ----------------------
def load_data(batch_size: int):
    """
    下载并加载 MNIST 数据。
    返回：训练集和测试集的 DataLoader。每个 batch 返回一组图像和对应标签。

    - transforms.ToTensor(): 将 PIL.Image 或 numpy.ndarray 转为 FloatTensor，且归一化到 [0.0, 1.0]
    - transforms.Normalize((mean,), (std,)): 标准化操作，将每个像素值按 (x-mean)/std

    MNIST 数据集包含 60000 张训练图片和 10000 张测试图片，灰度图，28x28 像素。
    """
    # MNIST 数据集的统计信息
    mean, std = 0.1307, 0.3081  # 计算方法：对整个训练集像素统计获得
    transform = transforms.Compose([
        transforms.ToTensor(),                    # 转为 [C,H,W]，且数值归一化到 [0,1]
        transforms.Normalize((mean,), (std,)),    # 标准化，使数据更服从正态分布，有助于加速模型收敛
    ])

    # train=True 下载训练集，train=False 下载测试集；download=True 表示若本地无数据则自动下载
    train_dataset = datasets.MNIST(root=DATA_DIR, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, transform=transform, download=True)

    # DataLoader 用于批量加载数据，shuffle=True 可在每个 epoch 之前打乱数据，更好地训练模型
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"每个 batch 大小: {batch_size}")
    return train_loader, test_loader

# ---------------------- 模型定义 ----------------------
class SimpleMLP(nn.Module):
    """
    两层全连接神经网络 (Multilayer Perceptron, MLP)：
    - 输入层：将 28x28 灰度图展开为 784 维向量
    - 隐藏层：全连接 + ReLU 激活
    - 输出层：全连接，输出 10 个类别的原始分数 (Logits)
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # nn.Linear(in_features, out_features)：创建一个全连接层
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 隐藏层：784 -> hidden_dim
        self.relu = nn.ReLU()                        # 非线性激活函数 ReLU
        self.fc2 = nn.Linear(hidden_dim, output_dim) # 输出层：hidden_dim -> 10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 初始形状 [batch_size, 1, 28, 28]
        # 使用 view() 将其变为 [batch_size, 784]
        x = x.view(x.size(0), -1)
        # fc1: 矩阵乘 + 偏置
        x = self.fc1(x)
        # 激活函数引入非线性
        x = self.relu(x)
        # fc2: 输出各类别的得分
        x = self.fc2(x)
        return x

# ---------------------- 训练函数 ----------------------
def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    device: torch.device,
                    epoch: int):
    """
    对模型进行一次完整的训练 (一个 epoch)。
    - 模型切换到训练模式 .train()
    - 对每个 batch：前向传播、计算损失、反向传播、参数更新
    - 每隔一定步数打印当前损失和计算时间
    """
    model.train()  # 启用 Dropout、BatchNorm 等训练时行为
    total_batches = len(dataloader)

    for batch_idx, (images, labels) in enumerate(dataloader, start=1):
        # 将数据拷贝到指定设备（CPU 或 GPU）
        images, labels = images.to(device), labels.to(device)

        start_time = time.time()  # 记录前向+反向计算时间

        # 前向传播: 通过模型得到预测值
        outputs = model(images)
        # nn.CrossEntropyLoss 内部包含 softmax + 负对数似然
        loss = criterion(outputs, labels)

        # 反向传播: 先清空梯度，再计算梯度，最后更新参数
        optimizer.zero_grad()  # 清空上一步残留的梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新模型参数

        elapsed_ms = (time.time() - start_time) * 1000  # 用毫秒为单位

        # 每 100 个 batch 或第 1 个 batch 打印日志，方便监控训练过程
        if batch_idx == 1 or batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}]"
                  f" Batch [{batch_idx}/{total_batches}]"
                  f" Loss: {loss.item():.4f}"
                  f" Time: {elapsed_ms:.1f} ms")

# ---------------------- 测试函数 ----------------------
def evaluate(model: nn.Module,
             dataloader: DataLoader,
             device: torch.device) -> None:
    """
    在测试集上评估模型性能：
    - 模型切换到评估模式 .eval(), 关闭梯度计算
    - 遍历所有测试样本，统计预测正确数量
    - 计算并打印整体准确率
    """
    model.eval()  # 关闭 Dropout 等训练时特有行为
    correct = 0  # 预测正确的样本总数
    total = 0    # 样本总数

    with torch.no_grad():  # 禁用梯度跟踪，提高推理速度并节省内存
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # torch.max 返回 (max_values, max_indices)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算准确率
    accuracy = 100.0 * correct / total
    print(f"测试集准确率: {accuracy:.2f}%\n")

# ---------------------- 主程序 ----------------------
def main():
    """
    1. 加载数据
    2. 创建模型、损失函数和优化器
    3. 进行多轮训练，并在每轮结束后评估
    4. 打印训练完成提示
    """
    print(f"使用设备: {DEVICE}")
    # 1. 加载训练集和测试集 DataLoader
    train_loader, test_loader = load_data(BATCH_SIZE)

    # 2. 实例化模型并移动到 GPU/CPU
    model = SimpleMLP(input_dim=28*28, hidden_dim=256, output_dim=10).to(DEVICE)
    print(model)
    # 交叉熵损失函数，常用于多分类任务
    criterion = nn.CrossEntropyLoss()
    # 随机梯度下降 (SGD) 优化器
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # 3. 训练和评估循环
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
        evaluate(model, test_loader, DEVICE)

    # 4. 完成提示
    print("训练完成！")

if __name__ == '__main__':
    main()
