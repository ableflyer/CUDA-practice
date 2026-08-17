#include <iostream>
#include <cuda_runtime.h>

__global__ void multVec(float *A, float *B, float *C, int n){
    int row = (blockIdx.x * blockDim.x) + threadIdx.x;
    int col = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (row < n && col < n){
        float sum = 0.0f;
        for(int k = 0; k < n; k++){
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main(){
    int n = 1 << 10;
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    size_t bytes = sizeof(float) * n * n;
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            h_a[i * n + j] = 2.0f;
            h_b[i * n + j] = 2.0f;
        }
    }

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    dim3 numThreads(16,16);
    dim3 numBlocks((n + 15) / 16, (n + 15) / 16);

    multVec<<<numBlocks, numThreads>>>(d_a, d_b, d_c, n);

    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        std::cerr << "Launch failed: " << cudaGetErrorString(launchErr) << std::endl;
    }

    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        std::cerr << "Sync failed: " << cudaGetErrorString(syncErr) << std::endl;
    }

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    bool correct = true;
    float expected = 4.0f * n;
    for (int i = 0; i < n * n; i++) {
        if (h_c[i] != expected) { correct = false; break; }
    }
    std::cout << (correct ? "correct\n" : "wrong\n");

    free(h_a);
    free(h_b);
    free(h_c);
    std::cout << "finished\n";

    return 0;
}