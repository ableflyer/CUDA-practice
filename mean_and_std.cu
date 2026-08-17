#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <math.h>

#define BLOCK_SIZE 256

using namespace std;
__global__ void reduceSum(float *input, float *output, int n){
    
    __shared__ float cache[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    cache[tid] = (i < n)? input[i]:0.0f;

    __syncthreads();

    for(int stride = blockDim.x/2; stride > 0; stride /= 2){
        if (tid < stride){
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0){
        output[blockIdx.x] = cache[0];
    }

}

__global__ void reduceSumSquaredDiff(float *input, float *output, float mean, int n){
    __shared__ float cache[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    cache[tid] = (i < n)? (input[i] - mean) * (input[i] - mean):0.0f;

    __syncthreads();

    for(int stride = blockDim.x/2; stride > 0; stride /= 2){
        if (tid < stride){
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0){
        output[blockIdx.x] = cache[0];
    }
}


int main(){
    srand(time(nullptr));
    int n = 1 << 20;
    size_t bytes = sizeof(float) * n;

    float* h_data = (float*) malloc(bytes);
    for (int i = 0; i < n; i++){
        h_data[i] = rand()/float(RAND_MAX) * 100.0f;
    }

    float* d_data;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float* d_partial;
    cudaMalloc(&d_partial, numBlocks * sizeof(float));
    reduceSum<<<numBlocks, BLOCK_SIZE>>>(d_data, d_partial, n);

    float *h_partial = (float*)malloc(numBlocks * sizeof(float));
    cudaMemcpy(h_partial, d_partial, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for(int i = 0; i < numBlocks; i++){
        sum += h_partial[i];
    }
    float mean = sum / n;

    reduceSumSquaredDiff<<<numBlocks, BLOCK_SIZE>>>(d_data, d_partial, mean, n);

    float *h_partial_2 = (float*)malloc(numBlocks * sizeof(float));
    cudaMemcpy(h_partial_2, d_partial, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    float sum_sq_dif = 0.0f;
    for(int i = 0; i < numBlocks; i++){
        sum_sq_dif += h_partial_2[i];
    }
    float variance = sum_sq_dif / n;
    float stddev = sqrtf(variance);

    cout << "GPU mean: " << mean << ", stddev: " << stddev << endl;

    float cpu_sum = 0.0f;
    for (int i = 0; i < n; i++) cpu_sum += h_data[i];
    float cpu_mean = cpu_sum / n;

    float cpu_sq = 0.0f;
    for (int i = 0; i < n; i++) cpu_sq += (h_data[i] - cpu_mean) * (h_data[i] - cpu_mean);
    float cpu_stddev = sqrtf(cpu_sq / n);

    std::cout << "CPU mean: " << cpu_mean << ", stddev: " << cpu_stddev << std::endl;

    cudaFree(d_data);
    cudaFree(d_partial);
    free(h_data);
    free(h_partial);
    free(h_partial_2);

    return 0;
}