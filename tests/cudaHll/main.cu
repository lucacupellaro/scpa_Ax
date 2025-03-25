#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "matriciOpp.h"
#include "../../lib/cuda/cudaHll.cu"



int main(int argc, char *argv[] ) {
   
    struct MatriceHLL *matHll;
    struct MatriceRaw *mat;
    struct Vector *vect;
    struct Vector *result;
    struct FlatELLMatrix *cudaHllMat;
    struct FlatELLMatrix*hMat;

    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s [matrix-market-filename] [hack]\n", argv[0]);
        exit(1);
    }

    int loadResult = loadMatRaw(argv[1], &mat);
    if (loadResult != 1)
    {
        printf("Errore leggendo la matrice\n");
        return 0;
    }

    int hack = atoi(argv[2]);
    printf("Hack size: %d\n", hack);

    int convResult = convertRawToHll(mat, hack, &matHll);
    if (convResult != 1)
    {
        printf("Error building HLL matrix, error code: %d\n", convResult);
        return convResult;
    }
   
   
    int vecResult = generate_random_vector(1, mat->width, &vect);
    if (vecResult != 0)
    {
        printf("Error while creating random vector\n");
        return vecResult;
    }

    int emptyResult = generateEmpty(mat->height, &result);
    if (emptyResult != 0)
    {
        printf("Error while creating result vector\n");
        return emptyResult;
    }

    int flatHll = convertHLLToFlatELL(&matHll, &cudaHllMat);
    if (emptyResult != 0)
    {
        printf("Error while converting to flat format result vector\n");
        return emptyResult;
    }

    printFlatELLMatrix(&cudaHllMat);

    int allocated = loadHLLFlatMatrixToGPUFromStruct(&cudaHllMat, &hMat);
    if (allocated != 0)
    {
        printf("Error while allocating on GPU\n");
        return emptyResult;
    }
    /*
    
    // Imposta una dimensione maggiore per il buffer di printf (opzionale)
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1048576);

    dim3 grid(matHll->numBlocks);
    dim3 block(matHll->HackSize);
    
    cudaMultiplyHLL<<<grid, block>>>(matHll, vect, result);
    cudaDeviceSynchronize();
    

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

    fflush(stdout);*/
    

    
    return 0;
}