#include<stdio.h>

int expFunc(int * array){
	printf("%d", array[1]);
	return 1;
}

int main(){
	int array[5] = {1,2,3,4,5};
	int * pointer;
	pointer = &array[0];
	expFunc(pointer);
}
