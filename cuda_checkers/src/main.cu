#include <iostream>
#include "kernel.h"

__global__ void myKernel() {
    // Kernel code goes here
}

int main() {
    // Launch the kernel
    myKernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel executed successfully!" << std::endl;
    return 0;
}