#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <time.h>
#include <stdlib.h>
//#include "CycleTimer.h"

__global__ void histogramKernel(int * inArray, int * outArray, int arrayLength, int significantDigit){

	int index 	= blockIdx.x * blockDim.x + threadIdx.x;

	int radix;
	int arrayElement;
	
	if(index < (arrayLength))
	{	
		arrayElement 		= inArray[index];
		radix			= ((arrayElement/significantDigit) % 10);
		
		atomicAdd(&outArray[radix], 1);
	}
}

__global__ void indexArrayKernel(int * inArray, int * bucketArray, int * indexArray, int arrayLength, int significantDigit){
	
	int index 	= blockIdx.x * blockDim.x + threadIdx.x;

	int i;
	int arrayElement;
	int radix;
	int pocket;
	
	if(index < 10){
		for(i = 0; i < arrayLength; i++){
	
			arrayElement 		= inArray[arrayLength-i-1];
			radix			= ((arrayElement/significantDigit) % 10);
			
			if(radix == index){
				pocket				= --bucketArray[radix];
				printf("\tArrayElement\t:\t%d\tbucketArray[%d]\t:\t%d\tPocket\t:\t%d\n", arrayElement, radix, bucketArray[radix],pocket);
				indexArray[arrayLength - i -1] 	= pocket;		
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
	printf("\n\nRunning Radix Sort on Unsorted List!\n\n");

	int significantDigit 	= 1;
	int largestNum 		= findLargestNum(array, size);

	int * inputArray;
	int * outputArray;
	int * bucketArray;
	int * semiSortArray;
	int * indexArray;

	int indexCheckArray[100000]	={0};

	while (largestNum / significantDigit > 0){
		printf("\tSorting: %d's place ", significantDigit);
		printArray(array, size);
		
		int bucket[10] = { 0 };
		int threadCount;
		int blockCount;
	
		threadCount 			= 256;
		blockCount 			= size/threadCount + 1;

		cudaMalloc((void **)& inputArray, sizeof(int)*size);
		cudaMalloc((void **)& indexArray, sizeof(int)*size);
		cudaMalloc((void **)& outputArray, sizeof(int)*size);
		cudaMalloc((void **)& semiSortArray, sizeof(int)*size);
		cudaMalloc((void **)& bucketArray, sizeof(int)*10);
		
		cudaMemcpy(inputArray, array, sizeof(int)*size, cudaMemcpyHostToDevice);
		cudaMemcpy(bucketArray, bucket, sizeof(int)*10, cudaMemcpyHostToDevice);
	 	
		histogramKernel<<<blockCount, threadCount>>>(inputArray, bucketArray, size, significantDigit); 	
		cudaThreadSynchronize();
		
		
		cudaMemcpy(bucket, bucketArray, sizeof(int)*10, cudaMemcpyDeviceToHost);
		printf("\tBucket Array");
		printArray(bucket, 10);
		

		cudaScanThrust(bucketArray, 10, bucketArray);	
		cudaMemcpy(bucket, bucketArray, sizeof(int)*10, cudaMemcpyDeviceToHost);
		printf("\tBucket Array");
		printArray(bucket, 10);
		
		indexArrayKernel<<<blockCount, threadCount>>>(inputArray, bucketArray, indexArray, size, significantDigit);
		cudaThreadSynchronize();
		cudaMemcpy(indexCheckArray, indexArray, sizeof(int)*size, cudaMemcpyDeviceToHost);
		printf("\tIndex Array");
		printArray(indexCheckArray, size);

		semiSortKernel<<<blockCount, threadCount>>>(inputArray, semiSortArray, indexArray, size, significantDigit);
		cudaThreadSynchronize();
		cudaMemcpy(array, semiSortArray, sizeof(int)*size, cudaMemcpyDeviceToHost);
		
		printf("\tSorted Array");
		printArray(array, size);

		significantDigit *= 10;

		printf("\n\tBucket: ");
		printArray(bucket, 10);

		cudaFree(inputArray);
		cudaFree(indexArray);
		cudaFree(bucketArray);
		cudaFree(outputArray);
		cudaFree(semiSortArray);

	}
}

int main(){

	printf("\n\nRunning Radix Sort Example in C!\n");
	printf("----------------------------------\n");

	int size = 100000;
	int* array;
	int i;
	int list[size];

	srand(time(NULL));

	for(i =0; i < size; i++){
		list[i]		= rand()%75;
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
