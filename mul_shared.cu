#include <iostream>
#include <cuda_runtime.h>

#define TILE 16

__global__ void multiply(float *A, float *B, float *C, int n){
    __shared__ float tileA[TILE][TILE], tileB[TILE][TILE];

    int row = (blockIdx.y * TILE) + threadIdx.y;
    int col = (blockIdx.x * TILE) + threadIdx.x;

    float sum = 0.0f;

    if(row < n && col < n){
        for(int t = 0; t < n/TILE; t++){
            tileA[threadIdx.y][threadIdx.x] = A[(row * n + (TILE * t + threadIdx.x))];
            tileB[threadIdx.y][threadIdx.x] = B[(TILE * t + threadIdx.y) * n + col];

            __syncthreads();

            for(int k = 0; k < TILE; k++){
                sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
            }
            __syncthreads();
        }
    }
    C[row * n + col] = sum;
}

int main(){
    int n = 1 << 10;
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    size_t bytes = sizeof(float) * n * n;

    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            h_a[i * n + j] = 2.0f;
            h_b[i * n + j] = 2.0f;
        }
    }

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    dim3 numThreads(16, 16);
    dim3 numBlocks((n + 15) / 16, (n + 15) / 16);

    multiply<<<numBlocks, numThreads>>>(d_a, d_b, d_c, n);
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