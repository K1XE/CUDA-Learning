import torch  # PyTorch 主库，用于张量操作和自动求导
import torch.nn as nn  # PyTorch 神经网络模块，用于定义模型结构
import time  # Python 标准库模块，用于测量时间
import polynomial_cuda  # 自定义编译好的 CUDA 扩展包，包含 polynomial_activation 函数


class CUDAPolynomialActivation(torch.autograd.Function):
    """
    自定义 Autograd Function，用于封装 CUDA 扩展的前向计算。
    继承自 torch.autograd.Function，需实现静态方法 forward 和 backward。
    """
    @staticmethod
    def forward(ctx, x):
        """
        前向计算，在这里调用我们编译好的 CUDA 扩展函数。
        参数:
            ctx: 上下文对象，用于在 forward 中保存变量供 backward 使用
            x: 输入张量，位于 GPU 上
        返回:
            扩展计算后的输出张量
        """
        # 直接调用扩展模块中的 polynomial_activation
        return polynomial_cuda.polynomial_activation(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播计算，需要实现才能支持自动求导。
        当前版本中未实现，调用会报错。
        参数:
            ctx: 上下文对象，可通过 ctx.saved_tensors 获取 forward 中保存的值
            grad_output: 上一层传下来的梯度张量
        返回:
            输入 x 的梯度张量（与 x 大小相同）
        """
        # 如果你要在训练中使用，请在此处实现多项式激活的导数：grad_input = grad_output * (2*x + 1)
        raise NotImplementedError("Backward pass not implemented")


class PolynomialActivation(nn.Module):
    """
    nn.Module 封装，使得多项式激活在模型中可以像普通层一样使用。
    支持两种实现：纯 PyTorch（自动求导）和 CUDA 扩展（前向高效）。
    """
    def __init__(self, implementation='pytorch'):
        """
        构造函数。
        参数:
            implementation: 'pytorch' 或 'cuda'，决定前向调用哪种实现
        """
        super().__init__()  # 调用父类构造函数
        self.implementation = implementation  # 保存用户选择的实现方式

    def forward(self, x):
        """
        前向计算分支。
        如果 implementation == 'pytorch'，使用 x**2 + x + 1；
        如果 implementation == 'cuda'，调用我们自定义的 Autograd Function。
        """
        if self.implementation == 'pytorch':
            # 纯 PyTorch 计算，多项式 x^2 + x + 1，支持自动求导
            return x**2 + x + 1
        elif self.implementation == 'cuda':
            # 调用封装好的 CUDA Function
            return CUDAPolynomialActivation.apply(x)
        else:
            # 非法参数检查
            raise ValueError(f"参数非法: {self.implementation}")


def benchmark(func, x, name, num_runs=1000):
    """
    性能基准测试函数。
    重复调用指定函数多次，并测量平均执行时间。

    参数:
        func: 待测函数，应接受一个张量并返回张量
        x: 输入张量，位于 GPU 上
        name: 测试名称，用于打印结果
        num_runs: 重复次数，默认 1000

    返回:
        格式化后的字符串，包含测试名称和平均耗时（ms）
    """
    # 记录起始时间（秒级）
    start_time = time.time()
    # 重复调用 func(x)
    for _ in range(num_runs):
        func(x)
    # 等待所有 CUDA 操作完成，确保测量精确
    torch.cuda.synchronize()
    # 记录结束时间
    end_time = time.time()
    # 计算平均耗时，并转换为毫秒
    avg_ms = (end_time - start_time) / num_runs * 1000
    # 返回格式化字符串
    return f"{name}: {avg_ms:.4f} ms"


def main():
    """
    主函数：生成数据、初始化层、预热、运行基准测试并打印结果。
    """
    # 固定随机种子，保证每次输入相同，便于对比
    torch.manual_seed(0)
    # 创建大小为 1e6 的随机张量，放到 GPU
    x = torch.randn(1000000, device='cuda')

    # 实例化两种实现并放到 GPU（虽然没有参数，但保持一致）
    pytorch_activation = PolynomialActivation(implementation='pytorch').cuda()
    cuda_activation = PolynomialActivation(implementation='cuda').cuda()

    # 预热：第一次调用可能包含加载/编译开销，此处调用并打印结果
    out = cuda_activation.forward(x)
    print("Sample output (first 10 elems):", out[:10])

    # 运行基准测试
    pytorch_time = benchmark(pytorch_activation, x, "PyTorch built-in")
    cuda_time = benchmark(cuda_activation,    x, "CUDA extension")

    # 打印基准结果
    print(pytorch_time)
    print(cuda_time)


if __name__ == "__main__":
    # 只有直接运行脚本时才调用 main()
    main()
