
#include <stdio.h>
#include <stdlib.h>
#include "matriciOpp.h"

int generate_random_vector(int seed, unsigned int righe, struct Vector **pointerToVector) {
    if (righe <= 0) {  // Controllo se righe è 0, dato che un vettore vuoto potrebbe non essere desiderato
        return -1;
    }

    // Alloca la struttura Vector e controlla il risultato
    *pointerToVector = malloc(sizeof(struct Vector));
    if (*pointerToVector == NULL) {
        return -1;
    }

    // Alloca l'array di double e controlla il risultato
    (*pointerToVector)->vettore = malloc(righe * sizeof(double));
    if ((*pointerToVector)->vettore == NULL) {
        free(*pointerToVector);
        return -1;
    }

    // Imposta correttamente il numero di elementi
    (*pointerToVector)->righr = righe;

    // Inizializza il generatore di numeri casuali con il seme fornito
    srand(seed);
    for (unsigned int i = 0; i < righe; i++) {
        (*pointerToVector)->vettore[i] = (double)rand() * RANDOM_VECTOR_MAX_VALUE / (RAND_MAX + 1.0);
    }
    
    return 0; // Successo
}


int freeRandom(struct Vector **pointerToVector){
    free((*pointerToVector)->vettore);
    free((*pointerToVector));
}


int generateEmpty(unsigned int rows, struct Vector **vettore) {
    *vettore = malloc(sizeof(struct Vector));
    if (*vettore == NULL) {
        return -1; 
    }
    
    (*vettore)->righr = rows;  
    (*vettore)->vettore = calloc(rows, sizeof(double));
    if ((*vettore)->vettore == NULL) {
        free(*vettore);
        return -1;  
    }
    
    return 0;  
}



void printVector(struct Vector *vec) {
    if (vec == NULL || vec->vettore == NULL) {
        printf("Invalid vector.\n");
        return;
    }

    printf("Vector with %u elements:\n", vec->righr);
    for (unsigned int i = 0; i < vec->righr; i++) {
        printf("%.4f ", vec->vettore[i]);
    }
    printf("\n");
}
