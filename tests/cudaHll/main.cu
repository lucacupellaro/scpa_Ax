#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "matriciOpp.h"
#include "../../lib/cuda/cudaHll.cu"




int main(int argc, char *argv[] ) {
   
    struct MatriceHLL *matHll;
    struct MatriceRaw *mat;
    struct Vector *vect;
    struct Vector *result;
    struct Vector *result2;
    struct Vector *resultSerial;
    struct FlatELLMatrix *cudaHllMat;
    cudaEvent_t start,stop;

    
  

    if (argc < 4)
    {
        fprintf(stderr, "Usage: %s [matrix-market-filename] [hack]\n", argv[0]);
        exit(1);
    }

    int loadResult = loadMatRaw(argv[1], &mat);
    if (loadResult != 1)
    {
        printf("Errore leggendo la matrice\n");
        return 0;
    }

    int hack = atoi(argv[2]);
    int k = atoi(argv[3]);
    

    
    
    
    int convResult = convertRawToHll(mat, hack, &matHll);
    if (convResult != 1)
    {
        printf("Error building HLL matrix, error code: %d\n", convResult);
        return convResult;
    }else{
        printf("convertita");
    }
   

   
    int vecResult = generate_random_vector(1, mat->width, &vect);
    if (vecResult != 0)
    {
        printf("Error while creating random vector\n");
        return vecResult;
    }


    int emptyResult = generateEmpty(mat->height, &result);
    if (emptyResult != 0)
    {
        printf("Error while creating result vector\n");
        return emptyResult;
    }

    emptyResult = generateEmpty(mat->height, &resultSerial);
    if (emptyResult != 0)
    {
        printf("Error while creating result vectorSerial\n");
        return emptyResult;
    }

    printf("\n dimensione resultSeires: %d",resultSerial->righe);

    

    emptyResult = generateEmpty(mat->height, &result2);
    if (emptyResult != 0)
    {
        printf("Error while creating result vectorSerial\n");
        return emptyResult;
    }

 

    int flatHll = convertHLLToFlatELL(&matHll, &cudaHllMat);
    if (emptyResult != 0)
    {
        printf("Error while converting to flat format result vector\n");
        return emptyResult;
    }

  

    int total_rows = 0;
    for (int i = 0; i < cudaHllMat->numBlocks; i++) {
        total_rows += cudaHllMat->block_rows[i];
    }

    if (k == 1) {
        printf("\n avvio kernel 1:\n");
        float gflop1 = invokeKernel1(vect, result, result2, resultSerial, cudaHllMat, matHll, hack);
        printf("GFLOPS Kernel 1: %lf\n", gflop1);
    } else if (k == 2) {
        printf("\n avvio kernel 2:\n");
        float gflop2 = invokeKernel2(vect, result, result2, resultSerial, cudaHllMat, matHll, hack);
        printf("GFLOPS Kernel 2: %lf\n", gflop2);
    } else if (k == 3) {
        printf("\n avvio kernel 3:\n");
        float gflop3 = invokeKernel3(vect, result, result2, resultSerial, cudaHllMat, matHll, hack);
        printf("GFLOPS Kernel 3: %lf\n", gflop3);
    } else {
        printf("valore non valido\n");
        exit(1); 
    }
    
   

    


    return 0;
}