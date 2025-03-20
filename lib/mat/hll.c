#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "matriciOpp.h"



int convertRawToHll(struct MatriceRaw *matricePointer, int hackSizeP, struct MatriceHLL **hllP) {
    int totalRows = matricePointer->height;
    int totalCols = matricePointer->width;
    int hackSize = hackSizeP;
    MatriceHLL *hll=malloc();
    *hllP=hll;
    int numBlocks = (totalRows + hackSize - 1) / hackSize; // arrotondamento per eccesso
    hll->totalRows = totalRows;
    hll->totalCols = totalCols;
    hll->HackSize = hackSize;
    hll->numBlocks = numBlocks;
    hll->blocks = malloc(numBlocks * sizeof(ELLPACK_Block));

    // Per ogni blocco...
    for (int blk = 0; blk < numBlocks; blk++) {
        // Creiamo una matrice temporanea RAW per il blocco corrente
        struct MatriceRaw blockRaw;
        blockRaw.width = matricePointer->width;
        blockRaw.height = hackSize;
        
        // Calcolare l'intervallo di righe
        int row_start = blk * hackSize;
        int row_end = (blk + 1) * hackSize;
        if (row_end > totalRows) row_end = totalRows;
        int blockRows = row_end - row_start;
        blockRaw.height = blockRows;

        // Selezionare i valori che appartengono a questo blocco
        int count = 0;
        for (int k = 0; k < matricePointer->nz; k++) {
            int i = matricePointer->iVettore[k];
            if (i >= row_start && i < row_end) {
                count++;
            }
        }
        blockRaw.nz = count;
        blockRaw.iVettore = malloc(count * sizeof(int));
        blockRaw.jVettore = malloc(count * sizeof(int));
        blockRaw.valori = malloc(count * sizeof(double));

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

        // Convertire il blocco RAW in ELLPACK_Block
        convertRawToEllpack(&blockRaw, blockRows, &hll->blocks[blk]);


        // Libero la memoria temporanea
        free(blockRaw.iVettore);
        free(blockRaw.jVettore);
        free(blockRaw.valori);
    }

    return 0; // successo
}

int convertRawToEllpack(struct MatriceRaw* matricePointer, int acksize, ELLPACK_Block* block) {
    int cols = matricePointer->width;
    int nz = matricePointer->nz;
    int blockRows = acksize; // acksize rappresenta il numero di righe effettive del blocco

    // Calcola il numero di non-zero per ogni riga del blocco
    int *row_nnz = calloc(blockRows, sizeof(int));
    for (int k = 0; k < nz; k++) {
        int i = matricePointer->iVettore[k]; // gli indici devono essere già locali (0...blockRows-1)
        row_nnz[i]++;
    }

    // Trova il massimo numero di non-zero per riga (MAXNZ)
    int MAXNZ = 0;
    for (int i = 0; i < blockRows; i++) {
        if (row_nnz[i] > MAXNZ) {
            MAXNZ = row_nnz[i];
        }
    }

    // Allocazione della struttura ELLPACK per il blocco
    block->M = blockRows;
    block->N = cols;
    block->MAXNZ = MAXNZ;
    block->JA = malloc(blockRows * sizeof(int *));
    block->AS = malloc(blockRows * sizeof(double *));
    for (int i = 0; i < blockRows; i++) {
        block->JA[i] = calloc(MAXNZ, sizeof(int));     // padding con zeri
        block->AS[i] = calloc(MAXNZ, sizeof(double));    // padding con zeri
    }

    // Riempie la struttura ELLPACK: per ogni riga, inserisce gli indici e i valori non-zero
    int *filled = calloc(blockRows, sizeof(int));
    for (int k = 0; k < nz; k++) {
        int i = matricePointer->iVettore[k]; // indice locale della riga (da 0 a blockRows-1)
        int j = matricePointer->jVettore[k];
        double val = matricePointer->valori[k];
        
        int pos = filled[i];               // posizione successiva libera nella riga i
        block->JA[i][pos] = j;
        block->AS[i][pos] = val;
        filled[i]++;
    }

    free(row_nnz);
    free(filled);
    return 0; // successo
}


