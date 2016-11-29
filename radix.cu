#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

__global__ void histogramKernel(int * inArray, int * outArray, int arrayLength, int pitSize, int significantDigit){
	
	int index 	= blockIdx.x * blockDim.x + threadIdx.x;

	int element;
	int arrayIndex	= index * pitSize;

	int i;

	if(arrayLength < (arrayIndex + pitSize - 1))
	{
		for(i = arrayIndex ; i < (arrayLength); i++){

			element		= ((inArray[i]/significantDigit) % 10);

			atomicInc(outArray[element], 1);
		}
	}
	else
	{
		for(i = arrayIndex ; i < (arrayIndex + pitSize); i++){

			element		= ((inArray[i]/significantDigit) % 10);

			atomicInc(outArray[element], 1);
		}
	}
}

__global__ void semiSortKernel(int * inArray, int * outArray, int* bucket){

	int index 	= blockIdx.x * blockDim.x + threadIdx.x;

	int element;
	int bucketIndex;
	int arrayIndex;

	if(index < arrayLength){
		element				= inArray[index];
		bucketIndex			= ((element/significantDigit) % 10);
		arrayIndex 			= atomicDec(bucket[bucketIndex] , 1);

		outArray[arrayIndex]		= element;
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

// double cudaScanThrust(int* inarray, int* end, int* resultarray) {

//     int length = end - inarray;
//     thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
//     thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
//     cudaMemcpy(d_input.get(), inarray, length * sizeof(int), 
//                cudaMemcpyHostToDevice);

//     double startTime = CycleTimer::currentSeconds();

//     thrust::exclusive_scan(d_input, d_input + length, d_output);

//     cudaThreadSynchronize();
//     double endTime = CycleTimer::currentSeconds();

//     cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
//                cudaMemcpyDeviceToHost);
//     thrust::device_free(d_input);
//     thrust::device_free(d_output);
//     double overallDuration = endTime - startTime;
//     return overallDuration;
// }


void cudaScanThrust(int* inarray, int arr_length, int* resultarray) {

    int length = arr_length;

    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), 
               cudaMemcpyHostToDevice);

    thrust::inclusive_scan(d_input, d_input + length, d_output);

    cudaThreadSynchronize();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
               cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
}

void radixSort(int * array, int size){
	printf("\n\nRunning Radix Sort on Unsorted List!\n\n");

	int i;
	int semiSorted[size];
	int significantDigit 	= 1;
	int largestNum 		= findLargestNum(array, size);
	int pitSize		= 4;

	while (largestNum / significantDigit > 0){
		printf("\tSorting: %d's place ", significantDigit);
		printArray(array, size);

		int bucket[10] = { 0 };

		histogramKernel<<<1, 256>>>(array, bucket, size, pitSize, significantDigit); 	

		cudaScanThrust(bucket, 10, bucket);	
					
		semiSortKernel<<<1, 256>>>(array, semiSorted, bucket)

		for (i = 0; i < size; i++)
			array[i] = semiSorted[i];

		significantDigit *= 10;

		printf("\n\tBucket: ");
		printArray(bucket, 10);

	}
}

int main(){

	printf("\n\nRunning Radix Sort Example in C!\n");
	printf("----------------------------------\n");

	int size = 12;

	int list[] = {10, 2, 303, 4021, 293, 1, 0, 429, 480, 92, 2999, 14};

	printf("\nUnsorted List: ");
	printArray(&list[0], size);

	radixSort(&list[0], size);

	printf("\nSorted List:");
	printArray(&list[0], size);

	printf("\n");

	return 0;
}