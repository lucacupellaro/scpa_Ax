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
                
                int idx = offset + j * rows + i;
                printf("[col=%d, val=%f] ", F->col_indices_flat[idx], F->values_flat[idx]);
            }
            printf("\n");
        }
        printf("\n");
    }
}



  

/*
__global__ void matvec_flatell_kernel(FlatELLMatrix *dMat, Vector *x, Vector *y, int total_rows)
{    
    
    
    int numBlocks = dMat->numBlocks;  // numero di blocchi

    int global_row = blockIdx.x * blockDim.x + threadIdx.x;

    int row_count = 0;
    int b = 0;
    while (b < numBlocks && (row_count + dMat->block_rows[b]) <= global_row) {
        row_count += dMat->block_rows[b];
        b++;
    }
    
    
    int local_row = global_row - row_count;
    int offset = dMat->block_offsets[b];
    int maxnz = dMat->block_nnz[b];
    int rows_in_block = dMat->block_rows[b]; //numero righe per blocco b

    

    double sum = 0.0;
    for (int j = 0; j < maxnz; j++) {
        int idx = offset + j * rows_in_block + local_row;
        int col = dMat->col_indices_flat[idx];
        
        if (col >= 0) {
            sum += dMat->values_flat[idx] * x->vettore[col];
        }
        
        
       
    }
    y->vettore[global_row] = sum;

     // Stampo direttamente (solo a scopo dimostrativo)
     //printf("Kernel: y[%d] = %f\n", global_row, sum);

    
    
}
*/



__global__ void matvec_flatell_kernel22(FlatELLMatrix *dMat, Vector *x, Vector *y, int total_rows) {  
    int numBlocks = dMat->numBlocks;  // Numero di blocchi
    int global_row = blockIdx.x * blockDim.x + threadIdx.x;  // Calcola l'indice della riga globale

    if (global_row >= total_rows) return;  // Se la riga globale è fuori dai limiti, termina il thread

    // Calcola il blocco senza ciclo
    int block_id = global_row / dMat->block_rows[0];  // Usa il numero di righe per blocco per determinare il blocco
    if (block_id >= numBlocks) return;  // Se il blocco supera il numero di blocchi, termina il thread

    int local_row = global_row % dMat->block_rows[block_id];  // Calcola la riga locale all'interno del blocco
    int offset = dMat->block_offsets[block_id];  // Ottieni l'offset del blocco corrente
    int maxnz = dMat->block_nnz[block_id];  // Numero massimo di elementi non nulli nel blocco
    int rows_in_block = dMat->block_rows[block_id];  // Numero di righe nel blocco

    double sum = 0.0;

    // Ciclo attraverso gli elementi non nulli del blocco
    for (int j = 0; j < maxnz; j++) {
        int idx = offset + j * rows_in_block + local_row;  // Calcola l'indice per la matrice piatta
        int col = dMat->col_indices_flat[idx];  // Ottieni la colonna corrispondente

        if (col >= 0) {  // Se la colonna è valida (non sparsa)
            sum += dMat->values_flat[idx] * x->vettore[col];  // Calcola il prodotto scalare
        }
    }

    // Scrivi il risultato nel vettore y
    y->vettore[global_row] = sum;

    // Stampa per debug (se necessario)
    //printf("Kernel: y[%d] = %f\n", global_row, sum);
}

__global__ void matvec_flatell_kernel2(FlatELLMatrix *dMat, Vector *x, Vector *y, int total_rows)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Calcola il numero totale di elementi non nulli potenziali (flatELL)
    int total_elements = 0;
    for (int i = 0; i < dMat->numBlocks; ++i) {
        total_elements += dMat->block_rows[i] * dMat->block_nnz[i];
    }

    if (tid >= total_elements) return;

    // Trova a quale blocco appartiene questo elemento
    int acc = 0;
    int b = 0;
    while (b < dMat->numBlocks) {
        int block_elems = dMat->block_rows[b] * dMat->block_nnz[b];
        if (tid < acc + block_elems) break;
        acc += block_elems;
        b++;
    }

    if (b >= dMat->numBlocks) return;

    // Calcoli locali all'interno del blocco b
    int local_tid = tid - acc;
    int rows_in_block = dMat->block_rows[b];
    int maxnz = dMat->block_nnz[b];
    int offset = dMat->block_offsets[b];

    int local_row = local_tid % rows_in_block;
    int j = local_tid / rows_in_block;

    if (j >= maxnz) return;

    int idx = offset + j * rows_in_block + local_row;
    int col = dMat->col_indices_flat[idx];

    if (col >= 0) {
        double product = dMat->values_flat[idx] * x->vettore[col];

        // Calcola la riga globale in base ai blocchi precedenti
        int global_row = 0;
        for (int k = 0; k < b; ++k) {
            global_row += dMat->block_rows[k];
        }
        global_row += local_row;

        atomicAdd(&y->vettore[global_row], product);
        //printf("Kernel: y[%d] = %f\n", global_row, product);
    }

    
}


