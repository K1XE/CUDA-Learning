#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

#define M 256
#define K 512
#define N 256
#define BLOCK_SIZE 16

__global__ void matMulKernel(float *A, float *B, float *C, int m, int k, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n)
    {
        float sums = 0.0f;
        for (int l = 0; l < k; l ++ )
        {
            sums += A[row * k + l] * B[l * n + col];
        }
        C[row * n + col] = sums;
    }
}

void matMul(float *A, float *B, float *C, int m, int k, int n)
{
    nvtxRangePush("MATRIX MULTIPLICATION");
    float *d_A, *d_B, *d_C;
    int sizeA = m * k * sizeof(float);
    int sizeB = k * n * sizeof(float);
    int sizeC = m * n * sizeof(float);

    nvtxRangePush("MEMORY ALLOCATION");
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    nvtxRangePop();

    nvtxRangePush("MEMORY COPY H2D");
    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
    nvtxRangePop();

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 num_block(
        (n + block_size.y - 1) / block_size.y,
        (m + block_size.x - 1) / block_size.x
    );

    nvtxRangePush("KERNEL EXECUTION");
    matMulKernel<<<num_block, block_size>>>(d_A, d_B, d_C, m, k, n);
    cudaDeviceSynchronize();
    nvtxRangePop();

    nvtxRangePush("MEMORY COPY D2H");
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
    nvtxRangePop();

    nvtxRangePush("MEMORY FREE");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    nvtxRangePop();

    nvtxRangePop();
}

void init_mat(float *mat, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i ++ )
    {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

int main(int argc, char** argv)
{
    float *h_A, *h_B, *h_C;
    size_t sizeA = M * K * sizeof(float),
           sizeB = K * N * sizeof(float),
           sizeC = M * N * sizeof(float);
    h_A = (float*)malloc(sizeA);
    h_B = (float*)malloc(sizeB);
    h_C = (float*)malloc(sizeC);

    init_mat(h_A, M, K);
    init_mat(h_B, K, N);

    matMul(h_A, h_B, h_C, M, K, N);

    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}