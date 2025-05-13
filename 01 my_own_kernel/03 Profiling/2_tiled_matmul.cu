#include <nvtx3/nvToolsExt.h>
#include <bits/stdc++.h>

#define TILE_SIZE 16
#define M 256
#define K 512
#define N 256


__global__ void matMulKernelOptimized(float *A, float *B, float *C, int m, int k, int n)
{
    __shared__ float shareA[TILE_SIZE][TILE_SIZE];
    __shared__ float shareB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x,  by = blockIdx.y,
        tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty,
        col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile ++ )
    {
        if (row < M && tile * TILE_SIZE + tx < K)
            shareA[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        else shareA[ty][tx] = 0.0f;
        if (col < N && tile * TILE_SIZE + ty < K)
            shareB[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        else shareB[ty][tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i ++ )
        {
            sum += shareA[ty][i] * shareB[i][tx];
        }

        __syncthreads();
    }
    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}

void matmul(float *A, float *B, float *C, int m, int k, int n)
{
    nvtxRangePush("MATRIX MULTIPLICATION");

    float *d_A, *d_B, *d_C;
    size_t sizeA = m * k * sizeof(float);
    size_t sizeB = k * n * sizeof(float);
    size_t sizeC = m * n * sizeof(float);

    nvtxRangePush("MEMORY ALLOCATION");
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    nvtxRangePop();

    nvtxRangePush("MEMORY COPY H2D");
    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
    nvtxRangePop();

    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 num_block(
        (N + block_size.y - 1) / block_size.y,
        (M + block_size.x - 1) / block_size.x
    );

    nvtxRangePush("KERNEL EXECUTION");
    matMulKernelOptimized<<<num_block, block_size>>>(d_A, d_B, d_C, M, K, N);
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

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    h_A = (float*)malloc(sizeA);
    h_B = (float*)malloc(sizeB);
    h_C = (float*)malloc(sizeC);

    init_mat(h_A, M, K);
    init_mat(h_B, K, N);

    matmul(h_A, h_B, h_C, M, K, N);

    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}