
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void addThreadId() 
{
	int i = threadIdx.x;


	printf("Hello world! My threadId is %d\n",i);
}

int main()
{
	const int threads = 256;
	addThreadId <<<1, threads >>> ();


	cudaDeviceSynchronize();
    return 0;
}