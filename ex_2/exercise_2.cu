
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <cstdlib>


void saxpyOnCpu(float* x, float* y, const float a, int arraySize);
void populateArrays(float* x, float* y, int arraySize);


__global__ void GPU_SAXPY(float *x,float *y, const float a,const int arraySize)
{

	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < arraySize) {
		y[i] = (a * x[i]) + y[i];
	}

}

int main()
{
	const int threads = 256;
	const int ARRAY_SIZE = 10000;
	const int threadBlocks = (ARRAY_SIZE + threads - 1) / threads;

	float x[ARRAY_SIZE] = {};
	float y[ARRAY_SIZE] = {};

	populateArrays(x, y, ARRAY_SIZE);
	const float A = 5;


	
	float *x_GPU = 0;
	float *y_GPU = 0;
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**)&x_GPU, ARRAY_SIZE * sizeof(float));
	cudaStatus = cudaMalloc((void**)&y_GPU, ARRAY_SIZE * sizeof(float));

	cudaStatus = cudaMemcpy(y_GPU, y, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(x_GPU, x, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);


	if (cudaStatus != cudaSuccess) {
		printf("malloc failed");
		goto FREE;
	}

	printf("Computing SAXPY on the CPU...");

	saxpyOnCpu(x, y, A, ARRAY_SIZE);

	printf("Done!\n\n");

	printf("Computing SAXPY on the GPU...");
	GPU_SAXPY <<<threadBlocks, threads >>> (x_GPU,y_GPU,A,ARRAY_SIZE);
	printf("Done!\n\n");
	cudaStatus = cudaDeviceSynchronize();

	float gpu_result[threadBlocks*threads] = {};


	cudaStatus = cudaMemcpy(gpu_result, y_GPU, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	if (cudaStatus != cudaSuccess)
	{
		goto FREE;
	}

	printf("Comparing the output for each implementation...");

	bool success = true;

	for (int i = 0; i < ARRAY_SIZE; i++) {
		if (abs(gpu_result[i] - y[i]) <= 0.00000001) {
			//We're good
		}
		else {
			printf("Failed! %f and %f are too different!", gpu_result[i], y[i]);
			success = false;

		}
	}

	if (success)
		printf("Correct!");




FREE:
	cudaFree(y_GPU);
	cudaFree(x_GPU);


	return 0;

}

void saxpyOnCpu(float* x, float* y, const float a,int arraySize) {

	for (int i = 0; i < arraySize; i++) {
		y[i] = a * x[i] + y[i];
	}
}



void populateArrays(float* x, float* y, int arraySize) {

	

	for (int i = 0; i < arraySize; i++) {
		x[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 10000));
		y[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 10000));
	}

}