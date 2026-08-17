#include <iostream>
#include <cuda_runtime.h>

// global is callable from the CPU and runs the GPU, but must be void, for we can't return with a GPU
__global__ void addVectors(float* A, float* B, float* C, int n){
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n){
        C[i] = A[i] + B[i];
    }
}

int main(){
    // 1 << 16 = 65536; written as a power-of-two shift for readability
    int n = 1 << 16;

    // setting up float pointers for Host(CPU) and Device(GPU) in order to communicate w/ each other later on in the code
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    // allocate the amount of bytes needed for CPU and GPU
    size_t bytes = sizeof(float) * n;

    // allocating and putting values to host
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // allocating device vars
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // sending the data a and b from host to device to compensate for the hardware isolation
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // setting up the threads and blocks, threads should be a multiple of 32 and numblocks should be fine when n changes
    int NumThreads = 256;
    int NumBlocks = (n + NumThreads - 1) / NumThreads;

    addVectors<<<NumBlocks, NumThreads>>>(d_a, d_b, d_c, n);

    // checking for errors if there are any
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        std::cerr << "Launch failed: " << cudaGetErrorString(launchErr) << std::endl;
    }

    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        std::cerr << "Sync failed: " << cudaGetErrorString(syncErr) << std::endl;
    }

    // sending the result from GPU to CPU, because the results are on GPU
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // we no longer need them, so they are free
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // checking if the calculation is correct
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (h_c[i] != 3.0f) { correct = false; break; }
    }
    std::cout << (correct ? "correct\n" : "wrong\n");

    // we no longer need these now, so they're also free
    free(h_a);
    free(h_b);
    free(h_c);
    std::cout << "finished\n";

    return 0;
}