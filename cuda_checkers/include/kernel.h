// This file declares the kernel functions and any necessary constants or types used in the CUDA project.

#ifndef KERNEL_H
#define KERNEL_H

// Function declarations for CUDA kernels
__global__ void myKernel(float *data, int size);

#endif // KERNEL_H