#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DEBUG 0

int convertHLLToFlatELL(MatriceHLL **H, FlatELLMatrix **flatMat)
{
    // Allocazione della struttura FlatELLMatrix
    *flatMat = (FlatELLMatrix *)malloc(sizeof(FlatELLMatrix));
    if (!(*flatMat))
    {
        perror("Errore di allocazione della struttura FlatELLMatrix");
        return -1;
    }

    int numBlocks = (*H)->numBlocks;
    (*flatMat)->numBlocks = numBlocks;

    // Calcola il numero totale di elementi da allocare
    int total = 0;
    for (int b = 0; b < numBlocks; b++)
    {
        ELLPACK_Block *block = (*H)->blocks[b];
        if (block)
        {
            total += block->M * block->MAXNZ;
        }
    }
    (*flatMat)->total_values = total;

    // Allocazione degli array per il formato Flat ELLPACK
    (*flatMat)->values_flat = (double *)malloc(total * sizeof(double));
    (*flatMat)->col_indices_flat = (int *)malloc(total * sizeof(int));
    (*flatMat)->block_offsets = (int *)malloc(numBlocks * sizeof(int));
    (*flatMat)->block_nnz = (int *)malloc(numBlocks * sizeof(int));
    (*flatMat)->block_rows = (int *)malloc(numBlocks * sizeof(int));

    if (!(*flatMat)->values_flat || !(*flatMat)->col_indices_flat ||
        !(*flatMat)->block_offsets || !(*flatMat)->block_nnz || !(*flatMat)->block_rows)
    {
        perror("Errore di allocazione negli array Flat ELLPACK");
        return -1;
    }

    int offset = 0;
    for (int b = 0; b < numBlocks; b++)
    {
        ELLPACK_Block *block = (*H)->blocks[b];
        if (!block)
            continue;

        int M = block->M;
        int MAXNZ = block->MAXNZ;

        // Salva i metadati per il blocco corrente
        (*flatMat)->block_offsets[b] = offset;
        (*flatMat)->block_nnz[b] = MAXNZ;
        (*flatMat)->block_rows[b] = M;

        // Copia dei dati: si copia in ordine colonna-per-riga.
        // L'elemento nella riga i e nella "colonna slot" j del blocco
        // viene memorizzato a: offset + j * M + i.
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < MAXNZ; j++)
            {
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

void printFlatELLMatrix(FlatELLMatrix **flatMat)
{
    if (flatMat == NULL || *flatMat == NULL)
    {
        printf("La struttura FlatELLMatrix è NULL.\n");
        return;
    }

    FlatELLMatrix *F = *flatMat;
    printf("Flat ELLPACK Matrix:\n");
    printf("Total values: %d, numBlocks: %d\n", F->total_values, F->numBlocks);

    // Scorre ciascun blocco
    for (int b = 0; b < F->numBlocks; b++)
    {
        int offset = F->block_offsets[b];
        int rows = F->block_rows[b];
        int maxnz = F->block_nnz[b];

        printf("Block %d: offset = %d, rows = %d, MAXNZ = %d\n", b, offset, rows, maxnz);

        // Per ogni riga del blocco
        for (int i = 0; i < rows; i++)
        {
            // Per ogni "slot" nella riga (fino a MAXNZ)
            for (int j = 0; j < maxnz; j++)
            {

                int idx = offset + j * rows + i;
                printf("[col=%d, val=%f] ", F->col_indices_flat[idx], F->values_flat[idx]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
__global__ void matvec_flatell_kernel(FlatELLMatrix *dMat, double *x, double *y, int hack_size) {
    int global_row = blockIdx.x * blockDim.x + threadIdx.x; 

    
    if (global_row >= dMat->numBlocks * hack_size) return;

    // Trova a quale blocco appartiene questa riga
    int block_id = global_row / hack_size;
    if (block_id >= dMat->numBlocks) return;

    int block_start = dMat->block_offsets[block_id];   // Offset del blocco
    int rows_in_block = dMat->block_rows[block_id];    // Righe nel blocco

    // Riga locale nel blocco
    int local_row = global_row % hack_size;
    if (local_row >= rows_in_block) return;

    double sum = 0.0;
    int max_nnz = dMat->block_nnz[block_id];  // NNZ massimo per riga nel blocco

    // Moltiplicazione matrice-vettore per la riga corrente
    for (int j = 0; j < max_nnz; j++) {

        int col = dMat->col_indices_flat[block_start + j * rows_in_block + local_row];
        if (col >= 0) {
            sum += dMat->values_flat[block_start + j * rows_in_block + local_row] * x[col];
        }
    }

    y[global_row] = sum;
}


__global__ void matvec_flatell_kernel2_reduction(FlatELLMatrix *dMat, double *x, double *y, int hack_size) {
    int global_row = blockIdx.x * blockDim.x + threadIdx.x;  // Indice globale della riga

    // Verifica se il thread è all'interno del range
    if (global_row >= dMat->numBlocks * hack_size) return;

    // Trova a quale blocco appartiene questa riga
    int block_id = global_row / hack_size;
    if (block_id >= dMat->numBlocks) return;

    int block_start = dMat->block_offsets[block_id];   // Offset del blocco
    int rows_in_block = dMat->block_rows[block_id];    // Righe nel blocco

    // Riga locale nel blocco
    int local_row = global_row % hack_size;
    if (local_row >= rows_in_block) return;

    // Memoria condivisa per la somma locale
    extern __shared__ double partial_sum[];  // Usa memoria condivisa dinamica, dimensione passata al kernel

    // Ogni thread inizia con una somma parziale
    double sum = 0.0;
    int max_nnz = dMat->block_nnz[block_id];  // NNZ massimo per riga nel blocco

    // Moltiplicazione matrice-vettore per la riga corrente
    for (int j = 0; j < max_nnz; j++) {
        int col = dMat->col_indices_flat[block_start + j * rows_in_block + local_row];
        if (col >= 0) {
            sum += dMat->values_flat[block_start + j * rows_in_block + local_row] * x[col];
        }
    }

    // Memorizziamo il risultato parziale nella memoria condivisa
    partial_sum[threadIdx.x] = sum;
    __syncthreads(); // Sincronizzazione di tutti i thread del blocco

    // Reduce dei risultati parziali all'interno del blocco
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + stride];
        }
        __syncthreads();  // Sincronizzazione di tutti i thread prima della successiva riduzione
    }

    // Il thread 0 scrive il risultato finale del blocco in y
    if (threadIdx.x == 0) {
        y[global_row] = partial_sum[0];
    }
}




__global__ void matvec_flatell_kernel3_safe(
    const double *values_flat,
    const int *col_indices_flat,
    const int *block_offsets,
    const int *block_nnz,
    const int *block_rows,
    const double *x,
    double *y,
    const int *numBlocks,  // Puntatore a int
    int hack_size,
    int x_length) 
{
    // Carica numBlocks dalla memoria globale
    const int nBlocks = *numBlocks;
    
    int global_row = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Verifica se il thread è all'interno del range
    if (global_row >= nBlocks * hack_size) return;

    // Trova a quale blocco appartiene questa riga
    int block_id = global_row / hack_size;
    if (block_id >= nBlocks) return;

    // Carica i metadati del blocco
    int block_start = block_offsets[block_id];
    int rows_in_block = block_rows[block_id];

    // Riga locale nel blocco
    int local_row = global_row % hack_size;
    if (local_row >= rows_in_block) return;

    double sum = 0.0;
    int max_nnz = block_nnz[block_id];  // NNZ massimo per riga nel blocco

    // Moltiplicazione matrice-vettore per la riga corrente
    for (int j = 0; j < max_nnz; j++) {
        int idx = block_start + j * rows_in_block + local_row;
        int col = col_indices_flat[idx];
        
        // Aggiunto controllo sui limiti di x rispetto alla versione originale
        if (col >= 0 && col < x_length) {
            sum += values_flat[idx] * x[col];
        }
    }

    // Scrittura del risultato
    y[global_row] = sum;
}