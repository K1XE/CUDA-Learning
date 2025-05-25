import torch
import triton
import triton.language as tl

@triton.jit
def addKernel(A, B, C, n_elems, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    masks = offsets < n_elems
    x = tl.load(A + offsets, mask=masks)
    y = tl.load(B + offsets, mask=masks)
    res = x + y
    tl.store(C + offsets, res, mask=masks)
    
def add(x: torch.tensor, y: torch.tensor):
    o = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and o.is_cuda, "Tensors must be on CUDA"
    n_elems = o.numel()
    grid = lambda meta : (triton.cdiv(n_elems, meta['BLOCK_SIZE']),)
    addKernel[grid](x, y, o, n_elems, 1024)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 28)],
        x_log=True,
        line_arg='provider',
        line_vals=['torch', 'triton'],
        line_names=['Torch', 'Triton'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='逐元素相加性能对比',
        args={}
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=[0.5, 0.2, 0.8])
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=[0.5, 0.2, 0.8])
    sz = 3 * x.numel() * x.element_size()
    gbps = lambda t: (sz * 1e-9) / (t * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, show_plots=True, save_path='.')
    