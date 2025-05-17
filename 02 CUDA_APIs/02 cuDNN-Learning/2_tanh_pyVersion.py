import torch
import time
import math
WARMUP_EPOCH = 10
BENCHMARK_EPOCH = 100
def my_tanh(x):
    return (torch.exp(2*x) - 1) / (torch.exp(2*x) + 1)

def benchmark_my_tanh(input_tensor):
    for _ in range(WARMUP_EPOCH):
        _ = my_tanh(input_tensor)
    
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(BENCHMARK_EPOCH):
        _ = my_tanh(input_tensor)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) * 1000 / BENCHMARK_EPOCH
    print(f"手写 Tanh: 平均 {avg_time:.3f} ms 完成一次")
    
    return my_tanh(input_tensor)

def benchmark_pytorch_tanh(input_tensor):
    for _ in range(BENCHMARK_EPOCH):
        _ = torch.tanh(input_tensor)
    
    start = time.perf_counter()
    for _ in range(BENCHMARK_EPOCH):
        _ = torch.tanh(input_tensor)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time = (end - start) * 1000 / BENCHMARK_EPOCH
    print(f"torch Tanh: 平均 {avg_time:.3f} ms 完成一次")
    
    return torch.tanh(input_tensor)

def verify_res(my_output, buildin_output):
    max_diff = torch.max(torch.abs(my_output - buildin_output)).item()
    print(f"二者最大误差为： {max_diff:.6e}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备为：{device}")

    input_tensor = torch.rand((256, 32, 128, 128), device=device) * 2 - 1

    _ = torch.tanh(input_tensor)
    torch.cuda.synchronize()

    my_output = benchmark_my_tanh(input_tensor)

    buildin_output = benchmark_pytorch_tanh(input_tensor)

    verify_res(my_output, buildin_output)

if __name__ == "__main__":
    main()