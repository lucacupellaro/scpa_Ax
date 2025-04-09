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

    
  

    if (argc < 3)
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
    printf("Hack size: %d\n", hack);

    
    
    
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

  
    //printFlatELLMatrix(&cudaHllMat);

    int total_rows = 0;
    for (int i = 0; i < cudaHllMat->numBlocks; i++) {
        total_rows += cudaHllMat->block_rows[i];
    }


    
    double *d_values_flat;
    cudaMalloc((void**)&d_values_flat, sizeof(double) * cudaHllMat->total_values);
    cudaMemcpy(d_values_flat, cudaHllMat->values_flat, sizeof(double) * cudaHllMat->total_values, cudaMemcpyHostToDevice);

    
    // Array degli indici di colonna flattenati
    int *d_col_indices_flat;
    cudaMalloc((void**)&d_col_indices_flat, sizeof(int) * cudaHllMat->total_values);
    cudaMemcpy(d_col_indices_flat, cudaHllMat->col_indices_flat, sizeof(int) * cudaHllMat->total_values, cudaMemcpyHostToDevice);

    

    // Array degli offset di inizio di ogni blocco
    int *d_block_offsets;
    cudaMalloc((void**)&d_block_offsets, sizeof(int) * cudaHllMat->numBlocks);
    cudaMemcpy(d_block_offsets, cudaHllMat->block_offsets, sizeof(int) * cudaHllMat->numBlocks, cudaMemcpyHostToDevice);

   

    // Array del numero massimo di non zero per riga per ogni blocco (MAXNZ)
    int *d_block_nnz;
    cudaMalloc((void**)&d_block_nnz, sizeof(int) * cudaHllMat->numBlocks);
    cudaMemcpy(d_block_nnz, cudaHllMat->block_nnz, sizeof(int) * cudaHllMat->numBlocks, cudaMemcpyHostToDevice);

    

    // Array del numero di righe effettive per ogni blocco
    int *d_block_rows;
    cudaMalloc((void**)&d_block_rows, sizeof(int) * cudaHllMat->numBlocks);
    cudaMemcpy(d_block_rows, cudaHllMat->block_rows, sizeof(int) * cudaHllMat->numBlocks, cudaMemcpyHostToDevice);

 

    
    // 2. Aggiornamento della struttura host con i puntatori device
    cudaHllMat->values_flat    = d_values_flat;
    cudaHllMat->col_indices_flat = d_col_indices_flat;
    cudaHllMat->block_offsets  = d_block_offsets;
    cudaHllMat->block_nnz      = d_block_nnz;
    cudaHllMat->block_rows     = d_block_rows;

    // 3. Allocazione della struttura sulla GPU e copia della struttura aggiornata
    struct FlatELLMatrix *d_mat;
    cudaMalloc((void**)&d_mat, sizeof(struct FlatELLMatrix));
    cudaMemcpy(d_mat, cudaHllMat, sizeof(struct FlatELLMatrix), cudaMemcpyHostToDevice);

    
  
    double *d_vettore;
    int righex=vect->righe;
    cudaMalloc((void**)&d_vettore, sizeof(double) * vect->righe);
    cudaMemcpy(d_vettore, vect->vettore, sizeof(double) * vect->righe, cudaMemcpyHostToDevice);

    // 2. Aggiorna il campo 'vettore' della struttura host per puntare all'array allocato sulla GPU
    double *temp=vect->vettore;
    vect->vettore = d_vettore;

    // 3. Alloca la struttura 'Vector' su GPU e copia la struttura aggiornata
    struct Vector *d_vect;
    cudaMalloc((void**)&d_vect, sizeof(struct Vector));
    cudaMemcpy(d_vect, vect, sizeof(struct Vector), cudaMemcpyHostToDevice);


    double *d_result_vettore;
    cudaMalloc((void**)&d_result_vettore, sizeof(double) * result->righe);
 

    // 2. Aggiorna il campo 'vettore' della struttura host per 'result'
    result->vettore = d_result_vettore;

    // 3. Alloca la struttura 'Vector' su GPU e copia la struttura aggiornata
    struct Vector *d_result;
    cudaMalloc((void**)&d_result, sizeof(struct Vector));
    cudaMemcpy(d_result, result, sizeof(struct Vector), cudaMemcpyHostToDevice);

   
   int *d_numBlocks;
    cudaMalloc(&d_numBlocks, sizeof(int));
    cudaMemcpy(d_numBlocks, &cudaHllMat->numBlocks, sizeof(int), cudaMemcpyHostToDevice);
   
    
   
   

    int block_size = 32;
    int num_threads = matHll->numBlocks * hack;
    int grid_size = (num_threads + block_size - 1) / block_size;

    printf("\nrighe: %d", righex);


    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    matvec_flatell_kernel2_reduction<<<grid_size, block_size,num_threads * sizeof(double)>>>(d_mat,d_vettore,d_result_vettore,hack);

    //matvec_flatell_kernel4<<<grid_size, block_size, sharedMemSize>>>(d_mat,d_vettore,d_result_vettore,hack,righex);

   /*
   matvec_flatell_kernel3_safe<<<grid_size, block_size>>>(
    d_values_flat,
    d_col_indices_flat,
    d_block_offsets,
    d_block_nnz,  
    d_block_rows,
    d_vettore,
    d_result_vettore,
    d_numBlocks,  
    hack,
    vect->righe);
   
   */

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);

    
    double time_sec = time_ms / 1000.0;

    double totalFLOPs = 2.0 * cudaHllMat->total_values;

    double gflops = totalFLOPs / (time_sec * 1e9);

    printf("Tempo medio del kernel: %f s\n", time_sec);
    printf("GFLOPS: %lf\n", gflops);


    cudaError memcopy;
     // Copia del risultato dalla GPU alla CPU
    memcopy=cudaMemcpy(result2->vettore, d_result_vettore, result2->righe * sizeof(double), cudaMemcpyDeviceToHost);
    if (memcopy!=cudaSuccess) {
        printf("errore");
    }   


    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_values_flat);
    cudaFree(d_col_indices_flat);
    cudaFree(d_vect);
    cudaFree(d_result);
    cudaFree(d_block_offsets);
    cudaFree(d_block_nnz);
    cudaFree(d_block_rows);

   
  
   
    vect->vettore=temp;
    
  
    
    double time2=0;

    int multResult = hllMultWithTime(&serialMultiplyHLL,matHll, vect, resultSerial, &time2);
    if (multResult != 0)
    {
        printf("Error in serialMultiply, error code: %d\n", multResult);
        return multResult;
    }
    


   for(int i=0;i < (result2->righe) ;i++){

        if(result2->vettore[i]!=resultSerial->vettore[i]){
            printf("\n: valori diversi (%lf vs %lf)\n", result2->vettore[i], resultSerial->vettore[i]);
        }

   }
    
    int check=areVectorsEqual(result2,resultSerial);
    if(check!=0){
        printf("the vector are different");
    }
    

    cudaError err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Errore nel lancio del kernel: %s\n", cudaGetErrorString(err));
        return -1;
    }

    
     
    

    return 0;
}