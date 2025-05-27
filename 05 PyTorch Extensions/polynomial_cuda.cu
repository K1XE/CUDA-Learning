#include <torch/extension.h> // PyTorch C++ API，包含 Tensor、构建扩展所需接口
#include <cuda.h>            // CUDA Driver API
#include <cuda_runtime.h>    // CUDA Runtime API，负责设备内存管理、流等

// ------------------------------------
// 2. CUDA Kernel：执行多项式运算
// ------------------------------------
// 模板参数 scalar_t 支持不同浮点类型（float、double）
// __global__ 标记函数为 GPU 内核
// 参数说明：
//   input  ：指向输入数据的指针（只读），加 __restrict__ 提示无别名
//   output ：指向输出数据的指针（可写），加 __restrict__ 提示无别名
//   length ：数据长度（元素个数）

template <typename scalar_t>
__global__ void polynomial_activation_kernel(
    const scalar_t *__restrict__ input,
    scalar_t *__restrict__ output,
    size_t length)
{
    // 计算该线程处理的全局元素索引
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查：确保 idx 在合法范围内
    if (idx < length)
    {
        // 从全局内存读取输入值
        const scalar_t x_val = input[idx];
        // 计算多项式 f(x) = x^2 + x + 1
        const scalar_t result = x_val * x_val + x_val + static_cast<scalar_t>(1);
        // 将结果写回输出数组
        output[idx] = result;
    }
}

// ------------------------------------
// 3. Host 函数：封装 CUDA Kernel 调用
// ------------------------------------
// 函数接受一个 CUDA Tensor，返回同形状的激活后结果

torch::Tensor polynomial_activation_cuda(const torch::Tensor &input_tensor)
{
    // 确保输入张量在 CUDA 设备上，否则抛出错误
    TORCH_CHECK(input_tensor.is_cuda(), "[polynomial_activation] Input must be a CUDA tensor");

    // 根据输入张量属性创建空输出张量（保留相同设备和类型）
    auto output_tensor = torch::empty_like(input_tensor);

    // 计算总元素数量
    const size_t num_elements = input_tensor.numel();

    // 配置 CUDA 执行参数：
    // threads_per_block：每个 Block 启动的线程数
    constexpr int threads_per_block = 1024;
    // num_blocks：所需的 Block 数量，向上取整
    const int num_blocks = static_cast<int>((num_elements + threads_per_block - 1) / threads_per_block);

    // 根据张量的数据类型动态选择 scalar_t
    AT_DISPATCH_FLOATING_TYPES(
        input_tensor.scalar_type(),   // 获取张量的 c10::ScalarType
        "polynomial_activation_cuda", // 宏标签，用于报错定位
        ([&]
         {
            // Kernel 调起：<<<grid, block>>>
            polynomial_activation_kernel<scalar_t>
                <<<num_blocks, threads_per_block>>>(
                    input_tensor.data_ptr<scalar_t>(),   // 获取输入指针
                    output_tensor.data_ptr<scalar_t>(),  // 获取输出指针
                    num_elements                         // 传入总元素数
                ); }));

    // 返回结果张量给 Python
    return output_tensor;
}

// ------------------------------------
// 4. PyBind11 模块：将函数暴露给 Python
// ------------------------------------
// TORCH_EXTENSION_NAME 由 setup.py 中定义

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // 定义 Python 接口函数名 "polynomial_activation"，绑定到 C++ 实现
    m.def(
        "polynomial_activation",                           // Python 调用名
        &polynomial_activation_cuda,                       // 对应的 C++ 函数
        "Polynomial activation (CUDA): f(x) = x^2 + x + 1" // Docstring
    );
}
