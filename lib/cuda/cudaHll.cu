#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WARP_SIZE 32

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


__global__ void matvec_flatell_kernel_2(FlatELLMatrix *dMat, double *x, double *y, int hack_size, int N) {
    extern __shared__ double shared_x[];
    int tid = threadIdx.x;
    int global_row = blockIdx.x * blockDim.x + tid;
    int block_size = blockDim.x;

    if (global_row >= dMat->numBlocks * hack_size) return;

    // Trova a quale blocco appartiene questa riga
    int block_id = global_row / hack_size;
    if (block_id >= dMat->numBlocks) return;

    int block_start = dMat->block_offsets[block_id];   // Offset del blocco
    int rows_in_block = dMat->block_rows[block_id];    // Righe nel blocco

    // Riga locale nel blocco
    int local_row = global_row % hack_size;
    if (local_row >= rows_in_block) return;

    // Caricamento di una porzione di x in memoria condivisa
    if (tid < block_size && tid < N) {
        shared_x[tid] = x[tid];
    }
    __syncthreads();

    double sum = 0.0;
    int max_nnz = dMat->block_nnz[block_id];  // NNZ massimo per riga nel blocco

    // Moltiplicazione matrice-vettore per la riga corrente
    for (int j = 0; j < max_nnz; j++) {
        int col = dMat->col_indices_flat[block_start + j * rows_in_block + local_row];
        if (col >= 0) {
            // Accesso a x dalla shared memory se l'indice è nel range caricato
            double x_val = (col < block_size && col < N) ? shared_x[col] : x[col];
            sum += dMat->values_flat[block_start + j * rows_in_block + local_row] * x_val;
        }
    }

    y[global_row] = sum;
}




__global__ void matvec_flatell_kernel_v4(FlatELLMatrix *dMat, double *x, double *y, int hack_size,int total_row) {

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;  // ID del thread
    int warp_id = thread_id >> 5;  // Ogni warp lavora su una riga (thread_id / 32)
    int lane = thread_id & 31;     // ID del thread dentro la warp (0-31)



    if (warp_id >= total_row) return;

    // Calcolare a quale hack appartiene questa riga (ogni hack corrisponde a un blocco)
    int block_id = warp_id / hack_size;
    int local_row = warp_id % hack_size;
    int rows_in_block = dMat->block_rows[block_id];

    if (local_row >= rows_in_block) return;  // Assicurarsi che non si esca dai limiti della riga

    int block_start = dMat->block_offsets[block_id];  // Offset del blocco
    int max_nnz_per_row = dMat->block_nnz[block_id]; // Max NNZ per riga nel blocco
    double sum = 0.0;

    for (int j = lane; j < max_nnz_per_row; j += 32) {
        
        int flat_idx = block_start + j * rows_in_block + local_row;

        int col = dMat->col_indices_flat[flat_idx];

        // Controlla se è un padding (spesso indicato con col < 0)
        if (col >= 0) {
            double val = dMat->values_flat[flat_idx];
            sum += val * x[col]; // Accumula il prodotto
        }
    }

    int width=32;
    // Riduzione a livello di warp per sommare i risultati parziali
    for (int offset = width >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset,width);
    }

    // Il primo thread della warp scrive il risultato finale
    if (lane == 0) {
        y[warp_id] = sum;
    }
}