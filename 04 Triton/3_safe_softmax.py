import torch
import triton
import triton.language as tl

@triton.jit
def safeSoftmaxKernel(
    outs, ins,
    outs_stride, ins_stride,
    cols, BLOCK_SIZE: tl.constexpr
):

    # 1.1 计算当前 program 负责第几行
    #    program_id(axis=0)      <=> CUDA 中的 blockIdx.x
    bid = tl.program_id(axis=0)
    
    # 1.2 计算这一行在内存中的起始指针
    #    input_ptr + row_idx * input_row_stride
    #    等价于在 CUDA 中：&input[row_idx][0]
    ins_start = ins + bid * ins_stride
    outs_start = outs + bid * outs_stride
    
    # ----------------------------------------
    # 2. 加载数据到 SRAM（Triton 的 vectorized load）
    # ----------------------------------------
    # tl.arange(0, BLOCK_SIZE) 生成一个 0..BLOCK_SIZE-1 的向量
    # mask 用来保证当 n_cols < BLOCK_SIZE 时不会越界读取
    # other=-inf 用来在超出范围的位置填充极小值，保证后面求 max 时不会受影响
    bsz = tl.load(ins_start + tl.arange(0, BLOCK_SIZE), 
                  mask=tl.arange(0, BLOCK_SIZE) < cols, 
                  other=float('-inf'))

    # ----------------------------------------
    # 3. 求当前行的最大值（numerical stability）
    # ----------------------------------------
    # tl.max(bsz, axis=0) 对向量 bsz 做归约求最大值
    maxV = tl.max(bsz, axis=0)
    
    # ----------------------------------------
    # 4. 减去最大值并做指数运算
    # ----------------------------------------
    # upper[i] = exp(bsz[i] - maxV)
    upper = tl.exp(bsz - maxV)
    
    # ----------------------------------------
    # 5. 求和，用于归一化
    # ----------------------------------------
    # tl.sum(upper, axis=0) 对向量做求和归约
    sums = tl.sum(upper, axis=0)
    
    # ----------------------------------------
    # 6. 归一化，得到 softmax 输出
    # ----------------------------------------
    softmax_output = upper / sums

    # ----------------------------------------
    # 7. 将结果写回到全局内存
    # ----------------------------------------
    # tl.store 功能与 tl.load 对应，mask 防止写出有效列以外的元素
    tl.store(outs_start + tl.arange(0, BLOCK_SIZE), 
             softmax_output, 
             mask=tl.arange(0, BLOCK_SIZE) < cols)

def triton_safe_softmax(x):
    # x: 一个形状为 [n_rows, n_cols] 的 CUDA Tensor
    n_rows, n_cols = x.shape
    # 为输出分配同大小同类型的 Tensor
    output = torch.empty_like(x)

    # 2.1 计算适合的 BLOCK_SIZE（列数对齐到 2 的幂，最大不超过 1024）
    #    Triton 建议使用 2 的幂次让 vectorization 最佳
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    
    # 2.2 配置 launch grid
    #    grid = (n_rows,) 等价于 CUDA 中的 <<<n_rows, BLOCK_SIZE>>>
    grid = (n_rows,)
    
    # 2.3 Launch Triton kernel
    safeSoftmaxKernel[grid](
        output,             # output_ptr
        x,                  # input_ptr
        output.stride(0),   # output_row_stride
        x.stride(0),        # input_row_stride （行跨度）
        n_cols,             # 列数
        BLOCK_SIZE=BLOCK_SIZE,  # 编译时常量
    )

    return output

# --------------------------------------------
# 3. 测试代码：对比 PyTorch 与 Triton 结果
# --------------------------------------------
torch.manual_seed(0)
x = torch.randn(256, 1024, device='cuda')

# 3.1 PyTorch 自带 softmax
torch_result = torch.softmax(x, dim=1)

# 3.2 Triton 版 softmax
triton_result = triton_safe_softmax(x)

# 3.3 计算最大误差
max_diff = torch.max(torch.abs(torch_result - triton_result))
print(f"pytorch与triton的safe-softmax最大误差为: {max_diff:.2e}")

# 3.4 检查是否在给定精度范围内
is_close = torch.allclose(torch_result, triton_result, rtol=1e-5, atol=1e-5)
print(f"结果正确否: {is_close}")