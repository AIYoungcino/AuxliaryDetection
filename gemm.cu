
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <ctime>
#include <iostream>
#define BLOCK_SIZE 256
__global__ void gpu_matrix_mult(int* a, int* b, int* c, int m, int n, int k)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int sum = 0;
	if (col < k && row < m)
	{
		for (int i = 0; i < n; i++)
		{
			sum += a[row * n + i] * b[i * k + col];
		}
		c[row * k + col] = sum;
	}
}

int main()
{
	int m=1024, n=1024, k=1024;
	srand(time(NULL));
	int* h_a, * h_b, * h_c, * h_cc;
	cudaMallocHost((void**)&h_a, sizeof(int) * m * n);
	cudaMallocHost((void**)&h_b, sizeof(int) * n * k);
	cudaMallocHost((void**)&h_c, sizeof(int) * m * k);
	cudaMallocHost((void**)&h_cc, sizeof(int) * m * k);
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			h_a[i * n + j] = rand() % 1024;
		}
	}
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < k; ++j) {
			h_b[i * k + j] = rand() % 1024;
		}
	}
	float gpu_elapsed_time_ms;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	int* d_a, * d_b, * d_c;
	cudaMalloc((void**)&d_a, sizeof(int) * m * n);
	cudaMalloc((void**)&d_b, sizeof(int) * n * k);
	cudaMalloc((void**)&d_c, sizeof(int) * m * k);
	cudaMemcpy(d_a, h_a, sizeof(int) * m * n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, sizeof(int) * n * k, cudaMemcpyHostToDevice);
	unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimGrid(grid_cols, grid_rows);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	gpu_matrix_mult << <dimGrid, dimBlock >> > (d_a, d_b, d_c, m, n, k);
	cudaMemcpy(h_c, d_c, sizeof(int) * m * k, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
	printf("Time elapsed on matrix multiplication of %dx%d.\n%dx%d on GPU: %f ms.\n\n", m, n, n, k, gpu_elapsed_time_ms);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_c);
	cudaFreeHost(h_cc);
	std::cin.get();
	return 0;
}
