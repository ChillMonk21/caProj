#include <stdio.h>

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

	

void radixSort(int * array, int size){
	printf("\n\nRunning Radix Sort on Unsorted List!\n\n");

	int i;
	int semiSorted[size];
	int significantDigit = 1;
	int largestNum = findLargestNum(array, size);

	while (largestNum / significantDigit > 0){
		printf("\tSorting: %d's place ", significantDigit);
		printArray(array, size);

		int bucket[10] = { 0 };

		for (i = 0; i < size; i++)
			bucket[(array[i] / significantDigit) % 10]++;

		for (i = 1; i < 10; i++)
			bucket[i] += bucket[i - 1];

		for (i = size - 1; i >= 0; i--){
	
			int index;
			int bucketIndex;
			int readValue;
			bucketIndex 		= (array[i] / significantDigit) % 10;
			readValue		= bucket[bucketIndex];
			index 			= (--bucket[bucketIndex]);
			
			printf("\t\tBucket Index\t:\t%d\n", bucketIndex);
			printf("\t\tRead Value\t:\t%d\n", readValue);
			printf("\t\tIndex\t\t:\t%d\n\n", index);

			semiSorted[index] = array[i];
		}

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