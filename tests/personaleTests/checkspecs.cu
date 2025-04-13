#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    // Get GPU device count
    int deviceCount;
    int result=cudaGetDeviceCount(&deviceCount);
printf("error %d\n",result);
    if (deviceCount == 0) {
        printf("No CUDA-compatible GPU found!\n");
        return 1;
    }
    deviceCount=1;
    // Loop through available devices
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("\n=== GPU Device %d: %s ===\n", dev, prop.name);
        printf("CUDA Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("Shared Memory per SM: %d KB\n", prop.sharedMemPerMultiprocessor / 1024);
        printf("Total Shared Memory per Block: %d KB\n", prop.sharedMemPerBlock / 1024);
        printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Threads per Warp: %d\n", prop.warpSize);
        printf("Max Grid Size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("Max Blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
        printf("Max Warps per SM: %d\n", prop.maxThreadsPerMultiProcessor / prop.warpSize);
        printf("Number of SMs: %d\n", prop.multiProcessorCount);
        printf("Clock Rate: %.2f MHz\n", prop.clockRate / 1000.0);
        printf("Memory Bus Width: %d-bit\n", prop.memoryBusWidth);
        printf("L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
    }

    return 0;
}
