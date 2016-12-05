#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <time.h>
#include <stdlib.h>

#include "CycleTimer.h"

#define SIZE 100000
#define THREADSIZE 256
#define BLOCKSIZE ((SIZE-1)/THREADSIZE + 1) 
#define RADIX 10

__global__ void copyKernel(int * inArray, int * semiSortArray, int arrayLength){
		
	int index 	= blockIdx.x * blockDim.x + threadIdx.x;
	
	if(index < arrayLength){
		inArray[index] 		= semiSortArray[index];
	}
}

__global__ void radixKernel(int * inArray, int* radixArray, int arrayLength, int significantDigit){

	__shared__ int inArrayShared[THREADSIZE];
	__shared__ int radixArrayShared[THREADSIZE];
	
	int index 	= blockIdx.x * blockDim.x + threadIdx.x;
	int thread 	= threadIdx.x;

	int arrayElement;
	int radix;

	if(index < arrayLength){
		inArrayShared[thread] 		= inArray[index];
	}
	
	if(index < arrayLength)
	{	
		arrayElement 			= inArrayShared[thread];
		radix				= ((arrayElement/significantDigit) % 10);
		radixArrayShared[thread]	= radix;
	}
	
	if(index < arrayLength){
		radixArray[index]		= radixArrayShared[thread];
	}

}
__global__ void histogramKernel(int * outArray, int * radixArray, int arrayLength, int significantDigit){

	__shared__ int outArrayShared[RADIX];
	
	int index 	= blockIdx.x * blockDim.x + threadIdx.x;
	int thread 	= threadIdx.x;
	int blockIndex	= blockIdx.x * RADIX;
	
	int radix;
	int i;

	if(thread ==  0){
		for(i =0; i < RADIX; i ++){
			outArrayShared[i] 		= 0;
		}
	}
	
	__syncthreads(); 

	if(index < arrayLength)
	{	
		radix				= radixArray[index];
		atomicAdd(&outArrayShared[radix], 1);
	}

	__syncthreads();
		

	if(thread == 0){
		for(i =0; i < RADIX; i++){
		
			outArray[blockIndex + i] 		= outArrayShared[i];
		}
	}
}

__global__ void combineBucket(int * blockBucketArray, int * bucketArray){
	
	__shared__ int bucketArrayShared[RADIX];

	int index 	= blockIdx.x * blockDim.x + threadIdx.x;
	
	int i;
	
	bucketArrayShared[index] 	= 0;
	
	for(i = index; i < RADIX*BLOCKSIZE; i=i+RADIX){
		atomicAdd(&bucketArrayShared[index], blockBucketArray[i]);		
	} 
	
	bucketArray[index] 		= bucketArrayShared[index];
}


__global__ void indexArrayKernel(int * radixArray,  int * bucketArray, int * indexArray, int arrayLength, int significantDigit){
	
	int index 	= blockIdx.x * blockDim.x + threadIdx.x;

	int i;
	int radix;
	int pocket;
	
	if(index < RADIX){
		
		for(i = 0; i < arrayLength; i++){
	
			radix			= radixArray[arrayLength -i -1];
			if(radix == index){
				pocket				= --bucketArray[radix];
				indexArray[arrayLength -i -1] 	= pocket;		
			}
		}
	}
}

__global__ void semiSortKernel(int * inArray, int * outArray, int* indexArray, int arrayLength, int significantDigit){

	int index 	= blockIdx.x * blockDim.x + threadIdx.x;
	
	int arrayElement;
	int arrayIndex;

	if(index < arrayLength){
		arrayElement			= inArray[index];
		arrayIndex 			= indexArray[index];
		outArray[arrayIndex]		= arrayElement;
	}
	
	

}

void printArray(int * array, int size){
	int i;
	printf("[ ");
	for (i = 0; i < size; i++)
		printf("%d ", array[i]);
	printf("]\n");
}

int findLargestNum(int * array, int size){
	int i;
	int largestNum = -1;
	for(i = 0; i < size; i++){
		if(array[i] > largestNum)
			largestNum = array[i];
	}
	return largestNum;
}


void cudaScanThrust(int* inarray, int arr_length, int* resultarray) {

    	int length = arr_length;
    
	thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    	thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    	cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    	thrust::inclusive_scan(d_input, d_input + length, d_output);

    	cudaThreadSynchronize();

    	cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    	thrust::device_free(d_input);
    	thrust::device_free(d_output);
}

void radixSort(int * array, int size){

	double startTime;
	double endTime;
	double duration;
	
	int significantDigit 	= 1;

	int threadCount;
	int blockCount;
	
	threadCount 			= THREADSIZE;
	blockCount 			= BLOCKSIZE;;
	
	int * outputArray;
	int * inputArray;
	int * radixArray;
	int * bucketArray;
	int * indexArray;
	int * semiSortArray;
	int * blockBucketArray;

	cudaMalloc((void **)& inputArray, sizeof(int)*size);
	cudaMalloc((void **)& indexArray, sizeof(int)*size);
	cudaMalloc((void **)& radixArray, sizeof(int)*size);
	cudaMalloc((void **)& outputArray, sizeof(int)*size);
	cudaMalloc((void **)& semiSortArray, sizeof(int)*size);
	cudaMalloc((void **)& bucketArray, sizeof(int)*RADIX);
	cudaMalloc((void **)& blockBucketArray, sizeof(int)*RADIX*BLOCKSIZE);	
	
	
	cudaMemcpy(inputArray, array, sizeof(int)*size, cudaMemcpyHostToDevice);
	
	int largestNum;
	thrust::device_ptr<int>d_in 	= thrust::device_pointer_cast(inputArray);
	thrust::device_ptr<int>d_out;
	d_out = thrust::max_element(d_in, d_in + size);
	largestNum	 	= *d_out;	
	printf("\tLargestNumThrust : %d\n", largestNum);
	
	startTime 	= CycleTimer::currentSeconds();	
	
	int displayArray[128];

	while (largestNum / significantDigit > 0){
	
		int bucket[RADIX] = { 0 };
		cudaMemcpy(bucketArray, bucket, sizeof(int)*RADIX, cudaMemcpyHostToDevice);
	 	
		radixKernel<<< blockCount, threadCount>>>(inputArray, radixArray, size, significantDigit);
		cudaThreadSynchronize();
			
		histogramKernel<<<blockCount, threadCount>>>(blockBucketArray, radixArray, size, significantDigit); 	
		cudaThreadSynchronize();
		cudaMemcpy(displayArray, blockBucketArray, sizeof(int)*20, cudaMemcpyDeviceToHost);
		printf("\nDisplayArray: ");
		printArray(displayArray, 20);
		
	
		combineBucket<<<1, RADIX>>>(blockBucketArray,bucketArray);
		cudaThreadSynchronize(); 			
		
		cudaScanThrust(bucketArray, RADIX, bucketArray);	
		cudaThreadSynchronize();
		
		indexArrayKernel<<<blockCount, threadCount>>>(radixArray, bucketArray, indexArray, size, significantDigit);
		cudaThreadSynchronize();

		semiSortKernel<<<blockCount, threadCount>>>(inputArray, semiSortArray, indexArray, size, significantDigit);
		cudaThreadSynchronize();
			
		copyKernel<<<blockCount, threadCount>>>(inputArray, semiSortArray, size);
		cudaThreadSynchronize();
		
		
		significantDigit *= RADIX;

	}
	
	endTime		= CycleTimer::currentSeconds();
	duration	= endTime - startTime;

	cudaMemcpy(array, semiSortArray, sizeof(int)*size, cudaMemcpyDeviceToHost);
	
	printf("Duration : %.3f ms\n", 1000.f * duration);
	
	cudaFree(inputArray);
	cudaFree(indexArray);
	cudaFree(radixArray);
	cudaFree(bucketArray);
	cudaFree(blockBucketArray);
	cudaFree(outputArray);
	cudaFree(semiSortArray);
}

int main(){

	printf("\n\nRunning Radix Sort Example in C!\n");
	printf("----------------------------------\n");

	int size = SIZE;
	int* array;
	int i;
	int list[size];

	srand(time(NULL));

	for(i =0; i < size; i++){
		list[i]		= SIZE -i;
	}
	
	array = &list[0];
	printf("\nUnsorted List: ");
	printArray(array, size);

	radixSort(array, size);

	printf("\nSorted List:");
	printArray(array, size);

	printf("\n");

	return 0;
}
