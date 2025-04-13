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
    //struct Vector *resultSerial;
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

    /*emptyResult = generateEmpty(mat->height, &resultSerial);
    if (emptyResult != 0)
    {
        printf("Error while creating result vectorSerial\n");
        return emptyResult;
    }

    printf("\n dimensione resultSeires: %d",resultSerial->righe);
*/
    

 
 

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

    double time=0;
    double totalFLOPs=0;
    double gflops=0;
    int result_=0;

    if (k == 1) {
        printf("\n avvio kernel 1:\n");
        result_ = invokeKernel1(vect, result, cudaHllMat, matHll, hack, &time);

        printf("tempo ritornato:\n %f",time);

        if(result_!=0){
            printf("kernel 1 crashed\n");
            exit;
        }
        totalFLOPs = 2.0 * cudaHllMat->total_values;
        gflops = totalFLOPs / (time * 1e9);
        printf("GFLOPS Kernel 1: %lf\n", gflops);
    } else if (k == 2) {
        printf("\n avvio kernel 2:\n");
        result_ = invokeKernel2(vect, result, cudaHllMat, matHll, hack,&time);
        if(result_!=0){
            printf("kernel 2 crashed");
            exit;
        }
        totalFLOPs = 2.0 * cudaHllMat->total_values;
        gflops = totalFLOPs / (time * 1e9);
        printf("GFLOPS Kernel 2: %lf\n", gflops);
    } else if (k == 3) {
        printf("\n avvio kernel 3:\n");
        result_ = invokeKernel3(vect, result, cudaHllMat, matHll, hack,&time);
        if(result_!=0){
            printf("kernel 3 crashed");
            exit;
        }
        totalFLOPs = 2.0 * cudaHllMat->total_values;
        gflops = totalFLOPs / (time * 1e9);
        printf("GFLOPS Kernel 3: %lf\n", gflops);
    } else {
        printf("valore non valido\n");
        exit(1); 
    }
    
   

    


    return 0;
}