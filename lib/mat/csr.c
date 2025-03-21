
#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "matriciOpp.h"
#include <time.h> 
typedef struct TempStruct{  // Defining a struct only for this function, for organizing the matrix in rows
    unsigned int elements;
 size_t bufferSize;
 unsigned int  *indices; // preparing a list of indexes of MatriceRaw format
};

int convertRawToCsr(struct MatriceRaw * matricePointer,struct MatriceCsr **csrPointer){ //Devo scorrere iVec e trovare tutti gli elementi di tutte le righe
    
  
    
    int nRighe=matricePointer->height;
    struct TempStruct *indicesArray=malloc(sizeof(struct TempStruct)*nRighe);// Each element rappresents a line in the matrix
    if(indicesArray==NULL)return-1;
    for(int i=0;i<nRighe;i++){
        indicesArray[i].elements=0;
        indicesArray[i].bufferSize=4;
        indicesArray[i].indices=malloc(indicesArray[i].bufferSize*sizeof(unsigned int));
        if(indicesArray[i].indices==NULL)return -1;
    }
    for(int i=0;i<matricePointer->nz;i++) {// per ogni riga aggiungo alla rispettiva struttura tutti gli indici di MatriceRaw che riguardano la riga stessa
        int indiceAttuale=matricePointer->iVettore[i];
        if(indicesArray[indiceAttuale].elements==indicesArray[indiceAttuale].bufferSize){
            unsigned int  *newindex=malloc(sizeof(unsigned int)*indicesArray[indiceAttuale].bufferSize*2);
            memcpy(newindex,indicesArray[indiceAttuale].indices,sizeof(unsigned int)*indicesArray[indiceAttuale].bufferSize);
            indicesArray[indiceAttuale].bufferSize=indicesArray[indiceAttuale].bufferSize*2; 
            free(indicesArray[indiceAttuale].indices);
            indicesArray[indiceAttuale].indices=newindex;
        }
        indicesArray[indiceAttuale].indices[indicesArray[indiceAttuale].elements]=i;
        indicesArray[indiceAttuale].elements+=1;
    }
    unsigned int numeroNz=matricePointer->nz;
    *csrPointer=malloc(sizeof(struct MatriceCsr));
    if(*csrPointer==NULL)return -1;
    struct MatriceCsr *matrixCsr=*csrPointer;
    matrixCsr->nz=numeroNz;
    matrixCsr->height=nRighe;
    matrixCsr->width=matricePointer->width;
    matrixCsr->iRP=malloc(sizeof(unsigned int)*(nRighe+1));
    matrixCsr->jValori=malloc(sizeof(unsigned int)*numeroNz);
    matrixCsr->valori=malloc(sizeof(double)*numeroNz);
    if(matrixCsr->iRP==NULL || matrixCsr->jValori==NULL || matrixCsr->valori==NULL)return -1;

    int j=0;
    matrixCsr->iRP[nRighe]=numeroNz;
    for(int i=0;i<nRighe;i++){
        matrixCsr->iRP[i]=j;
        for(int i2=0;i2<indicesArray[i].elements;i2++){
            matrixCsr->jValori[j]=matricePointer->jVettore[indicesArray[i].indices[i2]];
            matrixCsr->valori[j]=matricePointer->valori[indicesArray[i].indices[i2]];
            j++;
        }
    }

    for(int i=0;i<nRighe;i++){
        free(indicesArray[i].indices);
    }
    free(indicesArray);
    return 0;
}

int __attribute__((optimize("O0")))  serialCsrMult(struct MatriceCsr *csr,struct Vector *vec,struct Vector *result){
    unsigned int nrows=vec->righr;
    for(int i=0;i<nrows;i++){
        double sum=0;
        for(int j=csr->iRP[i];j<csr->iRP[i+1];j++){
            sum+=csr->valori[j]*vec->vettore[csr->jValori[j]];
        }
        result->vettore[i]=sum;
    }
}


#include <omp.h>
int __attribute__((optimize("O0"))) parallelCsrMult(struct MatriceCsr *csr, struct Vector *vec, struct Vector *result) {
    unsigned int nrows = vec->righr;

    #pragma omp parallel for
    for (int i = 0; i < nrows; i++) {
        double sum = 0.0;
        for (int j = csr->iRP[i]; j < csr->iRP[i + 1]; j++) {
            sum += csr->valori[j] * vec->vettore[csr->jValori[j]];
        }
        result->vettore[i] = sum;
    }
}


int csrMultWithTime(int (*multiplayer)(struct MatriceCsr *,struct Vector *,struct Vector *),struct MatriceCsr *csr,struct Vector *vec,struct Vector *result,double *execTime){
    clock_t t; 
    t = clock(); 
    multiplayer(csr,vec,result); 
    t = clock() - t; 
    (*execTime) = ((double)t)/CLOCKS_PER_SEC ; // in seconds 
}


int freeMatCsr(struct MatriceCsr ** matricePointer){
    free((*matricePointer)->iRP);
    free((*matricePointer)->jValori);
    free((*matricePointer)->valori);
    free(*matricePointer);
}


