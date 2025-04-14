
#include <stdio.h>
#include <stdlib.h>
#include "matriciOpp.h"
#include <float.h>

int areVectorsEqual( Vector *v1,  Vector *v2) {
    // Check if both vectors have the same number of rows
    if (v1->righe != v2->righe) {
        return -1;
    }

    // Compare each element within the defined tolerance (EPSILON)
    for (unsigned int i = 0; i < v1->righe; i++) {
        float module=(fabs(v1->vettore[i])+fabs(v2->vettore[i])/2);
        float diff=fabs(v1->vettore[i] - v2->vettore[i]);
        if(diff<EPSILON){
            continue;
        }else if(diff*100/(module)<0.02 ){
            continue;
        }else{
            printf("primo %.10f\nsecondo %.10f\n", v1->vettore[i],v2->vettore[i]);
            return -1;
        }
    }
    return 0;

   
}


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
    (*pointerToVector)->righe = righe;

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
    
    (*vettore)->righe = rows;  
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

    printf("Vector with %u elements:\n", vec->righe);
    for (unsigned int i = 0; i < vec->righe; i++) {
        printf("%.4f ", vec->vettore[i]);
    }
    printf("\n");
}
int calculate_vector_differences(const Vector *vec1, const Vector *vec2, double *sum_abs_diff, double *perc_diff)
{
    if (vec1 == NULL || vec2 == NULL || sum_abs_diff == NULL || perc_diff == NULL) {
        fprintf(stderr, "Error: NULL pointer passed to calculate_vector_differences.\n");
        if(sum_abs_diff) *sum_abs_diff = NAN;
        if(perc_diff) *perc_diff = NAN;
        return -1;
    }

    if (vec1->righe != vec2->righe) {
        fprintf(stderr, "Error: Vectors have different sizes (%u vs %u).\n", vec1->righe, vec2->righe);
        *sum_abs_diff = NAN;
        *perc_diff = NAN;
        return -1;
    }

    unsigned int n = vec1->righe;

     if (n > 0 && (vec1->vettore == NULL || vec2->vettore == NULL)) {
        fprintf(stderr, "Error: Internal vector data pointer is NULL for a non-empty vector.\n");
         *sum_abs_diff = NAN;
         *perc_diff = NAN;
        return -1;
    }

    if (n == 0) {
        *sum_abs_diff = 0.0;
        *perc_diff = 0.0;
        return 0;
    }

    double total_abs_diff = 0.0;
    double sum_abs_vec1 = 0.0;

    for (unsigned int i = 0; i < n; i++) {
        double diff = vec1->vettore[i] - vec2->vettore[i];
        total_abs_diff += fabs(diff);
        sum_abs_vec1 += fabs(vec1->vettore[i]);
    }

    *sum_abs_diff = total_abs_diff;

    if (fabs(sum_abs_vec1) < DBL_EPSILON) {
        if (fabs(total_abs_diff) < DBL_EPSILON) {
             *perc_diff = 0.0;
        } else {
            *perc_diff = NAN;
             fprintf(stderr, "Warning: Percentage difference is undefined (division by zero magnitude of vec1).\n");
        }
    } else {
        *perc_diff = (total_abs_diff / sum_abs_vec1) * 100.0;
    }

    return 0;
}

