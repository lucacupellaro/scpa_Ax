#ifdef USE_CUDA_ALLOC
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
// Definizione dei codici di errore
#define ERR_ALLOC_MATRICEHLL -1
#define ERR_ALLOC_BLOCKS -2
#define ERR_ALLOC_BLOCK -3
#define ERR_IDX_MISMATCH -4
#define ERR_ALLOC_ROWNNZ -5
#define ERR_ALLOC_FILLED -6
#define ERR_IDX_OUT_OF_RANGE -7

#define ALLOC(ptr, size)                                                        \
    do                                                                          \
    {                                                                           \
        fflush(stdout);                                                         \
        cudaError_t err = cudaMalloc((void **)&(ptr), (size));                  \
        if (err != cudaSuccess)                                                 \
        {                                                                       \
            fprintf(stderr, "cudaMalloc error for " #ptr ": %s (codice: %d)\n", \
                    cudaGetErrorString(err), err);                              \
            exit(1);                                                            \
        }                                                                       \
        fflush(stdout);                                                         \
    } while (0)
#define FREE(ptr) cudaFree(ptr)
#else
#include <stdlib.h>
#define ALLOC(ptr, size) (ptr = malloc(size))
#define FREE(ptr) free(ptr)
#endif
