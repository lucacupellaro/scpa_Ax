#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "matriciOpp.h"
#include <time.h> 
#include <omp.h>


int convertRawToHll(struct MatriceRaw *matricePointer, int hackSizeP, struct MatriceHLL **hllP) {
    int totalRows = matricePointer->height;
    int totalCols = matricePointer->width;
    int hackSize = hackSizeP;
    int numBlocks = (totalRows + hackSize - 1) / hackSize; // arrotondamento per eccesso

    printf("DEBUG: totalRows=%d, totalCols=%d, hackSize=%d, numBlocks=%d\n", totalRows, totalCols, hackSize, numBlocks);

    // Allocazione della struttura HLL
    struct MatriceHLL *hll = malloc(sizeof(struct MatriceHLL));
    if (hll == NULL) {
        fprintf(stderr, "Errore di allocazione per MatriceHLL.\n");
        return -1;
    }
    *hllP = hll;
    
    hll->totalRows = totalRows;
    hll->totalCols = totalCols;
    hll->HackSize = hackSize;
    hll->numBlocks = numBlocks;
    
    hll->blocks = malloc(numBlocks * sizeof(ELLPACK_Block *));
    if (hll->blocks == NULL) {
        fprintf(stderr, "Errore di allocazione per l'array dei blocchi.\n");
        free(hll);
        return -2;
    }
    for (int blk = 0; blk < numBlocks; blk++) {
        hll->blocks[blk] = malloc(sizeof(ELLPACK_Block));
        if (hll->blocks[blk] == NULL) {
            fprintf(stderr, "Errore di allocazione per il blocco %d.\n", blk);
            // Liberare quelli allocati finora e hll
            return -3;
        }
    }

    
    // Per ogni blocco...
    for (int blk = 0; blk < numBlocks; blk++) {
        printf("DEBUG: Elaborazione blocco %d\n", blk);
        struct MatriceRaw blockRaw;
        blockRaw.width = matricePointer->width;
        
        int row_start = blk * hackSize;
        int row_end = (blk + 1) * hackSize;
        if (row_end > totalRows) row_end = totalRows;
        int blockRows = row_end - row_start;
        blockRaw.height = blockRows;
        
        printf("DEBUG: Bloc %d, row_start=%d, row_end=%d, blockRows=%d\n", blk, row_start, row_end, blockRows);
        
        // Conta i valori appartenenti al blocco
        int count = 0;
        for (int k = 0; k < matricePointer->nz; k++) {
            int i = matricePointer->iVettore[k];
            if (i >= row_start && i < row_end) {
                count++;
            }
        }
        printf("DEBUG: Bloc %d, count non-zero elementi = %d\n", blk, count);
        
        blockRaw.nz = count;
        blockRaw.iVettore = malloc(count * sizeof(int));
        blockRaw.jVettore = malloc(count * sizeof(int));
        blockRaw.valori = malloc(count * sizeof(double));
        if (blockRaw.iVettore == NULL || blockRaw.jVettore == NULL || blockRaw.valori == NULL) {
            fprintf(stderr, "Errore di allocazione per il blocco RAW del blocco %d.\n", blk);
            free(blockRaw.iVettore);
            free(blockRaw.jVettore);
            free(blockRaw.valori);
            return -3;
        }
        
        int idx = 0;
        for (int k = 0; k < matricePointer->nz; k++) {
            int i = matricePointer->iVettore[k];
            if (i >= row_start && i < row_end) {
                blockRaw.iVettore[idx] = i - row_start; // shift per il blocco locale
                blockRaw.jVettore[idx] = matricePointer->jVettore[k];
                blockRaw.valori[idx] = matricePointer->valori[k];
                idx++;
            }
        }
        if (idx != count) {
            fprintf(stderr, "Errore: idx (%d) diverso da count (%d) nel blocco %d.\n", idx, count, blk);
            return -4;
        }
        printf("DEBUG: Bloc %d, blockRaw.nz = %d\n", blk, blockRaw.nz);
        
        // Converti il blocco RAW in ELLPACK_Block
        int err = convertRawToEllpack(&blockRaw, blockRows, &hll->blocks[blk]);
        if (err != 0) {
            fprintf(stderr, "Errore nella conversione del blocco %d: codice %d\n", blk, err);
            return err;
        }
        printf("DEBUG: Bloc %d, conversione in ELLPACK completata.\n", blk);
        
        free(blockRaw.iVettore);
        free(blockRaw.jVettore);
        free(blockRaw.valori);
    }
    
    return 1; // Successo
}



int convertRawToEllpack(struct MatriceRaw* matricePointer, int acksize, ELLPACK_Block** block_) {
    int cols = matricePointer->width;
    int nz = matricePointer->nz;
    int blockRows = acksize; // acksize rappresenta il numero di righe effettive del blocco

    printf("DEBUG: cols=%d, nz=%d, blockRows=%d, \n", cols, nz, blockRows);
    // Dereferenzia il puntatore al blocco
    ELLPACK_Block *block = *block_;

    // Calcola il numero di non-zero per ogni riga del blocco
    int *row_nnz = calloc(blockRows, sizeof(int));
    if (row_nnz == NULL) {
        fprintf(stderr, "Errore di allocazione per row_nnz.\n");
        return -1;
    }
    for (int k = 0; k < nz; k++) {
        int i = matricePointer->iVettore[k]; // gli indici devono essere locali al blocco, cioè in [0, blockRows)
        if (i < 0 || i >= blockRows) {
            fprintf(stderr, "DEBUG ERROR: indice %d fuori dal range [0, %d) in posizione k=%d\n", i, blockRows, k);
            free(row_nnz);
            return -1;
        }
        row_nnz[i]++;
    }

    // Trova il massimo numero di non-zero per riga (MAXNZ)
    int MAXNZ = 0;
    for (int i = 0; i < blockRows; i++) {
        if (row_nnz[i] > MAXNZ) {
            MAXNZ = row_nnz[i];
        }
    }

    printf("DEBUG: MAXNZ=%d\n",  MAXNZ);
    // Se non ci sono elementi, MAXNZ sarà 0; potresti voler gestire questo caso se necessario.
    
    // Allocazione della struttura ELLPACK per il blocco
    block->M = blockRows;
    block->N = cols;
    block->MAXNZ = MAXNZ;

    printf("DEBUG:N blocchi block=%d\n",  block->M);
 
    
    block->JA = malloc(blockRows * sizeof(int *));
    block->AS = malloc(blockRows * sizeof(double *));
    if (block->JA == NULL || block->AS == NULL) {
        fprintf(stderr, "Errore di allocazione per JA o AS.\n");
        free(row_nnz);
        return -1;
    }
    for (int i = 0; i < blockRows; i++) {
        block->JA[i] = calloc(MAXNZ, sizeof(int));     // padding con zeri (o -1 se preferisci)
        block->AS[i] = calloc(MAXNZ, sizeof(double));    // padding con zeri
        if (block->JA[i] == NULL || block->AS[i] == NULL) {
            fprintf(stderr, "Errore di allocazione per riga %d.\n", i);
            free(row_nnz);
            return -1;
        }
    }
    
    // Riempie la struttura ELLPACK: per ogni riga, inserisce gli indici e i valori non-zero
    int *filled = calloc(blockRows, sizeof(int));
    if (filled == NULL) {
        fprintf(stderr, "Errore di allocazione per filled.\n");
        free(row_nnz);
        return -1;
    }
    for (int k = 0; k < nz; k++) {
        int i = matricePointer->iVettore[k]; // indice locale della riga (deve essere in [0, blockRows))
        if (i < 0 || i >= blockRows) {
            fprintf(stderr, "DEBUG ERROR: indice %d fuori dal range [0, %d) (fase riempimento) in posizione k=%d\n", i, blockRows, k);
            free(row_nnz);
            free(filled);
            return -1;
        }
        int pos = filled[i];  // posizione successiva libera nella riga i
        if (pos < 0 || pos >= MAXNZ) {
            fprintf(stderr, "DEBUG ERROR: pos %d fuori dal range [0, %d) per riga %d\n", pos, MAXNZ, i);
            free(row_nnz);
            free(filled);
            return -1;
        }
        int j = matricePointer->jVettore[k];
        double val = matricePointer->valori[k];
        
        block->JA[i][pos] = j;
        block->AS[i][pos] = val;
        filled[i]++;
    }
    
    free(row_nnz);
    free(filled);
    return 0; // successo
}



int printHLL(struct MatriceHLL **hllP) {
    if (hllP == NULL || *hllP == NULL) {
        fprintf(stderr, "La struttura HLL è NULL.\n");
        return -1;
    }
    
    MatriceHLL *hll = *hllP;
    printf("Matrice HLL:\n");
    printf("  Total Rows: %d\n", hll->totalRows);
    printf("  Total Cols: %d\n", hll->totalCols);
    printf("  HackSize: %d\n", hll->HackSize);
    printf("  Num Blocks: %d\n", hll->numBlocks);
    
    // Per ogni blocco della matrice HLL
    for (int blk = 0; blk < hll->numBlocks; blk++) {
        printf("\nBlock %d:\n", blk);
        ELLPACK_Block *block = hll->blocks[blk];
        printf("  M (righe): %d\n", block->M);
        printf("  N (colonne): %d\n", block->N);
        printf("  MAXNZ (max non-zero per riga): %d\n", block->MAXNZ);
        
        // Per ogni riga del blocco, stampo gli array JA e AS
        for (int i = 0; i < block->M; i++) {
            printf("    Row %d:\n", i);
            printf("      JA: ");
            for (int j = 0; j < block->MAXNZ; j++) {
                printf("%d ", block->JA[i][j]);
            }
            printf("\n");
            
            printf("      AS: ");
            for (int j = 0; j < block->MAXNZ; j++) {
                printf("%f ", block->AS[i][j]);
            }
            printf("\n");
        }
    }
    
    return 0;
}

int serialMultiply(struct MatriceHLL** matriceHLL, struct Vector *vec, struct Vector *result){



    if (!matriceHLL || !(*matriceHLL) || !vec || !result)
        return -1;

    struct MatriceHLL *mat = *matriceHLL;

    if (vec->righr != mat->totalCols || result->righr != mat->totalRows)
        return -1;

    for (int b = 0; b < mat->numBlocks; b++) {
        ELLPACK_Block *block = mat->blocks[b];
        int globalRowStart = b * mat->HackSize;

        for (int i = 0; i < block->M; i++) {
            double t = 0.0;
            for (int j = 0; j < block->MAXNZ; j++) {
                int colIndex = block->JA[i][j];
                double val = block->AS[i][j];

                // CONTROLLO INDISPENSABILE
                if(colIndex >= 0 && colIndex < vec->righr) {
                    t += val * vec->vettore[colIndex];
                }
            }
            result->vettore[globalRowStart + i] = t;
        }
    }

    return 0; // esecuzione corretta
}

int serialHllMultWithTime(struct MatriceHLL **HllMat,struct Vector *vec,struct Vector *result,double *execTime){
  
    clock_t t; 
    t = clock(); 
    serialMultiply(HllMat,vec,result); 
    t = clock() - t; 
    (*execTime) = ((double)t)/CLOCKS_PER_SEC ; // in seconds 
    return 0;
}


int openMpHllMultWithTime(struct MatriceHLL **HllMat,struct Vector *vec,struct Vector *result,double *execTime){
  
    clock_t t; 
    t = clock(); 
    openMpMultiply(HllMat,vec,result); 
    t = clock() - t; 
    (*execTime) = ((double)t)/CLOCKS_PER_SEC ; // in seconds 
    return 0;
}


int openMpMultiply(struct MatriceHLL** matriceHLL, struct Vector *vec, struct Vector *result) {

    if (!matriceHLL || !(*matriceHLL) || !vec || !result)
        return -1;

    struct MatriceHLL *mat = *matriceHLL;

    if (vec->righr != mat->totalCols || result->righr != mat->totalRows)
        return -1;
   
    omp_set_num_threads(20);
    #pragma pragma omp parallel(static)
   
    for (int b = 0; b < mat->numBlocks; b++) {
        ELLPACK_Block *block = mat->blocks[b];
        int globalRowStart = b * mat->HackSize;

        int t=omp_get_max_threads();
        //printf("n threads: %d\n", t);

        
        for (int i = 0; i < block->M; i++) {
            double t = 0.0;
            for (int j = 0; j < block->MAXNZ; j++) {
                int colIndex = block->JA[i][j];
                double val = block->AS[i][j];

                // Controllo indispensabile: assicura che l'indice sia valido
                if (colIndex >= 0 && colIndex < vec->righr) {
                    t += val * vec->vettore[colIndex];
                }
            }
            result->vettore[globalRowStart + i] = t;
        }
    }

    return 0; // esecuzione corretta
}




