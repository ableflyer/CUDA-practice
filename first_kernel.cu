#include <iostream>
#include <cuda_runtime.h>

// CUDA Kernel function to execute on the GPU
__global__ void helloFromGPU() {
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

int main() {
    std::cout << "Hello World from CPU!" << std::endl;

    // Launch kernel with 1 block and 5 threads
    helloFromGPU<<<1, 5>>>();

    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        std::cerr << "Launch failed: " << cudaGetErrorString(launchErr) << std::endl;
    }

    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        std::cerr << "Sync failed: " << cudaGetErrorString(syncErr) << std::endl;
    }

    // Wait for the GPU to finish before exiting
    cudaDeviceSynchronize();

    return 0;
}