#include <stdio.h>
#include <cuda.h>
#define CONSTANT_MEM 64*1024


__constant__ char constValue[CONSTANT_MEM];

__global__ void dkernel(){ //__global__ indicate it is not normal kernel function but for GPU
printf("Hello world \n");
}




int main (){

int size=64*1024
dkernel <<<1,2>>>();//<<<no. of blocks,no. of threads in in block>>>

cudaDeviceSynchronize(); //Tells GPU to do all work than synchronize GPU buffer with CPU.

return 0;

}