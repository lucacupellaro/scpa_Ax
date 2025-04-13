#include <cuda_runtime.h>
#include <stdio.h>

__global__ void helloKernel() {
    printf("Ciao da CUDA thread %d\n", threadIdx.x);
}

int main() {
    printf("Starting kernel...\n");

    // Imposta una dimensione maggiore per il buffer di printf (opzionale)
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1048576);

    // Lancia il kernel con 1 blocco di 10 thread
    helloKernel<<<1, 10>>>();

    // Controlla subito errori di lancio
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    // Sincronizza e controlla errori
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error after synchronization: %s\n", cudaGetErrorString(err));
    }

    // Forza lo svuotamento degli output
    fflush(stdout);

    printf("Kernel execution finished.\n");
    return 0;
}