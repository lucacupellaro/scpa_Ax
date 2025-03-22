#include <stdio.h>
#include <stdlib.h>
#include "matriciOpp.h"

int main(int argc, char *argv[]) {

    struct MatriceHLL *matHll;
    struct MatriceRaw *mat;
    struct Vector* vect;
    struct Vector* result;

    if (argc < 3) {
        fprintf(stderr, "Usage: %s [matrix-market-filename] [hack]\n", argv[0]);
        exit(1);
    }

    int loadResult = loadMatRaw(argv[1], &mat);
    if (loadResult != 1) {
        printf("Errore leggendo la matrice\n");
        return 0;
    }

    fprintf(stdout, "nz=%d height=%d width=%d\n", mat->nz, mat->height, mat->width);
    for(int i = 0; i < mat->nz; i++) {
        fprintf(stdout, "%d %d %20.19g\n", mat->iVettore[i]+1, mat->jVettore[i]+1, mat->valori[i]);
    }

    int hack = atoi(argv[2]);
    printf("Hack size: %d\n", hack);

    int convResult = convertRawToHll(mat, hack, &matHll);
    if (convResult != 1) {
        printf("Error building HLL matrix, error code: %d\n", convResult);
        return convResult;
    }

    /*int printResult = printHLL(&matHll);
    if (printResult != 0) {
        printf("Error while reading HLL matrix\n");
        return printResult;
    }*/

    int vecResult = generate_random_vector(1, mat->width, &vect);
    if (vecResult != 0) {
        printf("Error while creating random vector\n");
        return vecResult;
    }

    int emptyResult = generateEmpty(mat->height, &result);
    if (emptyResult != 0) {
        printf("Error while creating result vector\n");
        return emptyResult;
    }

    printf("Input vector:\n");
    printVector(vect);

    double time=0;
    double time2=0;

    int multResult = serialHllMultWithTime(&matHll, vect, result,&time);
    //int multResult = serialMultiply(&matHll, vect, result);
    if (multResult != 0) {
        printf("Error in serialMultiply, error code: %d\n", multResult);
        return multResult;
    }

    printf("Serial calculation for nz:%u,%f time, %f GFLOPS",mat->nz,time,2.0*mat->nz/(time*1000000000));

    printf("Result vector (y = Ax) Serial:\n");
    //printVector(result);

    //freeRandom(&vect);
    //freeRandom(&result);

    //int multResult2 = openMpMultiply(&matHll, vect, result);
    int multResult2 = openMpHllMultWithTime(&matHll, vect, result,&time2);
    if (multResult != 0) {
        printf("Error in serialMultiply, error code: %d\n", multResult2);
        return multResult;
    }
    printf("OpenMp calculation for nz:%u,%f time, %f GFLOPS",mat->nz,time2,2.0*mat->nz/(time2*1000000000));

    printf("Result vector (y = Ax) OpenMP:\n");
    //printVector(result);

    freeRandom(&vect);
    freeRandom(&result);

    return 0;
}