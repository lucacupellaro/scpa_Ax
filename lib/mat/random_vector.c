
#include <stdio.h>
#include <stdlib.h>
#include "matriciOpp.h"

int generate_random_vector(int seed, unsigned int righe, struct Vector **pointerToVector){
    
    if (righe <= 0) {
        return -1;
    }
    *pointerToVector=malloc(sizeof(struct Vector));
    (*pointerToVector)->vettore= (double *)malloc(righe * sizeof(double));
    if (*pointerToVector == NULL) {
        return -1;
    }
    (*pointerToVector)->righr=righe;
    srand(seed);
    for (int i = 0; i < righe; i++) {
        (*pointerToVector)->vettore[i] = (double)rand()*RANDOM_VECTOR_MAX_VALUE / (RAND_MAX + 1.0) ; // Generates a number in (0,1)
    }
    
    return 0; // Success
}

int freeRandom(struct Vector **pointerToVector){
    free((*pointerToVector)->vettore);
    free((*pointerToVector));
}

int generateEmpty(unsigned int rows,struct Vector **vettore){
    (*vettore)=malloc(sizeof(struct Vector));
    (*vettore)->righr=rows;
    (*vettore)->vettore=calloc(sizeof(double),rows);
}

printf("ciao");

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
