import os
import numpy as np
from torchvision import datasets, transforms

# 1. 创建目标保存目录
#    save_dir: MNIST 数据保存路径，若不存在则创建
save_dir = "mnist_data"
os.makedirs(save_dir, exist_ok=True)

# 2. 定义数据预处理操作
#    transforms.ToTensor(): 将 PIL 图像或 numpy.ndarray 转为 [0,1] 范围的 Tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# 3. 下载并加载 MNIST 数据集
#    train=True: 加载训练集；train=False: 加载测试集
#    root="./data": 临时下载目录
mnist_train = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
mnist_test = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# 4. 提取数据和标签，并展开图像为一维向量
#    mnist_train.data: 形状 (60000, 28, 28)，uint8，像素值 [0,255]
#    reshape(-1, 28*28): 展平为 (60000, 784)
#    astype(np.float32): 转为 float32 类型
#    /255.0: 归一化到 [0,1]
X_train = mnist_train.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
#    mnist_train.targets: 训练标签，形状 (60000,)
y_train = mnist_train.targets.numpy().astype(np.int32)

#    同理处理测试集
X_test = mnist_test.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
y_test = mnist_test.targets.numpy().astype(np.int32)

# 5. 将处理后的数据保存为二进制文件 (little-endian)
#    .tofile: 直接将 ndarray 原始内存写入文件
X_train.tofile(os.path.join(save_dir, "X_train.bin"))
y_train.tofile(os.path.join(save_dir, "y_train.bin"))
X_test.tofile(os.path.join(save_dir, "X_test.bin"))
y_test.tofile(os.path.join(save_dir, "y_test.bin"))

# 6. 写入元数据说明文件，方便后续读取和使用
#    metadata.txt 包含样本数、输入维度、类别数等信息
with open(os.path.join(save_dir, "metadata.txt"), "w") as f:
    f.write(f"Training samples: {X_train.shape[0]}\n")           # 训练样本数量
    f.write(f"Test samples: {X_test.shape[0]}\n")               # 测试样本数量
    f.write(f"Input dimensions: {X_train.shape[1]}\n")         # 每个样本特征长度
    f.write(f"Number of classes: {len(np.unique(y_train))}\n") # 类别数量

# 7. 完成提示
print("MNIST 数据集已下载并以二进制格式保存至目录：", save_dir)
