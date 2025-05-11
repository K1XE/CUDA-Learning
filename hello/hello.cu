#include <stdio.h>
__global__ void myKernel(void)
{
    printf("HELLO CUDA !!!\n");
}

int main(int argc, char **argv)
{
    dim3 b_of_g(2, 2, 2), t_of_b(1, 1, 2);

    myKernel<<<b_of_g, t_of_b>>>();

    cudaDeviceSynchronize();

    return 0;
}