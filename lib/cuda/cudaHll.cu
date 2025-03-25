#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define DEBUG 0


int convertHLLToFlatELL(MatriceHLL** H, FlatELLMatrix** flatMat) {
    // Allocazione della struttura FlatELLMatrix
    *flatMat = (FlatELLMatrix*)malloc(sizeof(FlatELLMatrix));
    if (!(*flatMat)) {
        perror("Errore di allocazione della struttura FlatELLMatrix");
        return -1;
    }
    
    int numBlocks = (*H)->numBlocks;
    (*flatMat)->numBlocks = numBlocks;
    
    // Calcola il numero totale di elementi da allocare
    int total = 0;
    for (int b = 0; b < numBlocks; b++) {
        ELLPACK_Block* block = (*H)->blocks[b];
        if (block) {
            total += block->M * block->MAXNZ;
        }
    }
    (*flatMat)->total_values = total;
    
    // Allocazione degli array per il formato Flat ELLPACK
    (*flatMat)->values_flat = (double*)malloc(total * sizeof(double));
    (*flatMat)->col_indices_flat = (int*)malloc(total * sizeof(int));
    (*flatMat)->block_offsets = (int*)malloc(numBlocks * sizeof(int));
    (*flatMat)->block_nnz = (int*)malloc(numBlocks * sizeof(int));
    (*flatMat)->block_rows = (int*)malloc(numBlocks * sizeof(int));
    
    if (!(*flatMat)->values_flat || !(*flatMat)->col_indices_flat ||
        !(*flatMat)->block_offsets || !(*flatMat)->block_nnz || !(*flatMat)->block_rows) {
        perror("Errore di allocazione negli array Flat ELLPACK");
        return -1;
    }
    
    int offset = 0;
    for (int b = 0; b < numBlocks; b++) {
        ELLPACK_Block* block = (*H)->blocks[b];
        if (!block) continue;
        
        int M = block->M;
        int MAXNZ = block->MAXNZ;
        
        // Salva i metadati per il blocco corrente
        (*flatMat)->block_offsets[b] = offset;
        (*flatMat)->block_nnz[b] = MAXNZ;
        (*flatMat)->block_rows[b] = M;
        
        // Copia dei dati: si copia in ordine colonna-per-riga.
        // L'elemento nella riga i e nella "colonna slot" j del blocco
        // viene memorizzato a: offset + j * M + i.
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < MAXNZ; j++) {
                int dst_idx = offset + j * M + i;
                int src_idx = i * MAXNZ + j; // Gli array JA e AS sono in ordine riga-per-riga
                (*flatMat)->values_flat[dst_idx] = block->AS[src_idx];
                (*flatMat)->col_indices_flat[dst_idx] = block->JA[src_idx];
            }
        }
        offset += M * MAXNZ;
    }
    
    return 0;
}


void printFlatELLMatrix(FlatELLMatrix** flatMat) {
    if (flatMat == NULL || *flatMat == NULL) {
        printf("La struttura FlatELLMatrix è NULL.\n");
        return;
    }
    
    FlatELLMatrix *F = *flatMat;
    printf("Flat ELLPACK Matrix:\n");
    printf("Total values: %d, numBlocks: %d\n", F->total_values, F->numBlocks);
    
    // Scorre ciascun blocco
    for (int b = 0; b < F->numBlocks; b++) {
        int offset = F->block_offsets[b];
        int rows   = F->block_rows[b];
        int maxnz  = F->block_nnz[b];
        
        printf("Block %d: offset = %d, rows = %d, MAXNZ = %d\n", b, offset, rows, maxnz);
        
        // Per ogni riga del blocco
        for (int i = 0; i < rows; i++) {
            // Per ogni "slot" nella riga (fino a MAXNZ)
            for (int j = 0; j < maxnz; j++) {
                // Gli elementi sono memorizzati in ordine colonna-per-riga:
                // Indice = offset + j * rows + i
                int idx = offset + j * rows + i;
                printf("[col=%d, val=%f] ", F->col_indices_flat[idx], F->values_flat[idx]);
            }
            printf("\n");
        }
        printf("\n");
    }
}


int loadHLLFlatMatrixToGPUFromStruct(FlatELLMatrix **hMat, FlatELLMatrix **dMat) {
    cudaError_t err;
    
    // Allocazione della struttura device (su host) per conservare i puntatori GPU
    *dMat = (FlatELLMatrix *)malloc(sizeof(FlatELLMatrix));
    if (*dMat == NULL) {
        fprintf(stderr, "Errore nell'allocazione della struttura dMat\n");
        return -1;
    }
    
    // Copia dei metadati dalla struttura host a quella device (la struttura dMat)
    (*dMat)->total_values = (*hMat)->total_values;
    (*dMat)->numBlocks    = (*hMat)->numBlocks;
    
    // Allocazione della memoria device per ciascun campo
    
    // Allocazione per i valori flattenati
    err = cudaMalloc((void**)&((*dMat)->values_flat), (*hMat)->total_values * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Errore in cudaMalloc per dMat->values_flat: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Allocazione per gli indici di colonna flattenati
    err = cudaMalloc((void**)&((*dMat)->col_indices_flat), (*hMat)->total_values * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Errore in cudaMalloc per dMat->col_indices_flat: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Allocazione per gli offset di ogni blocco
    err = cudaMalloc((void**)&((*dMat)->block_offsets), (*hMat)->numBlocks * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Errore in cudaMalloc per dMat->block_offsets: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Allocazione per il MAXNZ per ogni blocco
    err = cudaMalloc((void**)&((*dMat)->block_nnz), (*hMat)->numBlocks * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Errore in cudaMalloc per dMat->block_nnz: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Allocazione per il numero di righe per ogni blocco
    err = cudaMalloc((void**)&((*dMat)->block_rows), (*hMat)->numBlocks * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Errore in cudaMalloc per dMat->block_rows: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Copia dei dati dalla memoria host a quella device per ciascun array
    
    err = cudaMemcpy((*dMat)->values_flat, (*hMat)->values_flat, (*hMat)->total_values * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Errore in cudaMemcpy per dMat->values_flat: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    err = cudaMemcpy((*dMat)->col_indices_flat, (*hMat)->col_indices_flat, (*hMat)->total_values * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Errore in cudaMemcpy per dMat->col_indices_flat: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    err = cudaMemcpy((*dMat)->block_offsets, (*hMat)->block_offsets, (*hMat)->numBlocks * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Errore in cudaMemcpy per dMat->block_offsets: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    err = cudaMemcpy((*dMat)->block_nnz, (*hMat)->block_nnz, (*hMat)->numBlocks * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Errore in cudaMemcpy per dMat->block_nnz: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    err = cudaMemcpy((*dMat)->block_rows, (*hMat)->block_rows, (*hMat)->numBlocks * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Errore in cudaMemcpy per dMat->block_rows: %s\n", cudaGetErrorString(err));
        return -1;
    }

    double *d_ptr = NULL;               // Dichiarazione della variabile per il puntatore device
    size_t size = 1024 * sizeof(double);  // Dimensione in byte dell'array da allocare

  err = cudaMalloc((void**)&d_ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Errore in cudaMalloc: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("Allocazione riuscita, d_ptr = %p\n", d_ptr);
    
    return 0;
}

__global__ void matvec_flatell_kernel(
    const double* __restrict__ values,
    const int* __restrict__ col_indices,
    const int* __restrict__ block_offsets,
    const int* __restrict__ block_nnz,
    const int* __restrict__ block_rows,
    const double* __restrict__ x,  // vettore di input
    double* y,                     // vettore risultato
    int numBlocks,
    int total_rows)
{
    int global_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_row >= total_rows) return;

    // Determina a quale blocco appartiene la riga globale.
    // Si assume che i blocchi siano concatenati in output: le prime block_rows[0] righe appartengono al blocco 0, poi block_rows[1] e così via.
    int row_count = 0;
    int b = 0;
    while (b < numBlocks && (row_count + block_rows[b]) <= global_row) {
        row_count += block_rows[b];
        b++;
    }
    if (b >= numBlocks) return; // fuori range
    int local_row = global_row - row_count;

    int offset = block_offsets[b];
    int maxnz = block_nnz[b];
    int rows_in_block = block_rows[b];

    double sum = 0.0;
    // Per ogni "slot" della riga (fino a maxnz), accedi al valore e all'indice di colonna
    for (int j = 0; j < maxnz; j++) {
        int idx = offset + j * rows_in_block + local_row;
        int col = col_indices[idx];
        if (col >= 0) {
            sum += values[idx] * x[col];
        }
    }
    y[global_row] = sum;
}