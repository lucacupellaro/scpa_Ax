
#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "matriciOpp.h"

typedef struct TempStruct{
    unsigned int elements;
    unsigned int bufferSize;
    unsigned int  *indices;
};

int convertRawToCsr(struct MatriceRaw * matricePointer,struct MatriceCsr **csrPointer){ //Devo scorrere iVec e trovare tutti gli elementi di tutte le righe
    int nRighe=matricePointer->height;
    struct TempStruct *indicesArray=malloc(sizeof(struct TempStruct)*nRighe);
    for(int i=0;i<nRighe;i++){
        indicesArray[i].elements=0;
        indicesArray[i].bufferSize=4;
        indicesArray[i].indices=malloc(sizeof(unsigned int)*4);
    }
    for(int i=0;i<matricePointer->nz;i++) {
        int indiceAttuale=matricePointer->iVettore[i];
        if(indicesArray[indiceAttuale].elements==indicesArray[indiceAttuale].bufferSize){
            indicesArray[indiceAttuale].bufferSize=indicesArray[indiceAttuale].bufferSize*2; 
            indicesArray[indiceAttuale].indices=realloc(indicesArray[indiceAttuale].indices,indicesArray[indiceAttuale].bufferSize);
        }
        indicesArray[indiceAttuale].indices[indicesArray[indiceAttuale].elements]=i;
        indicesArray[indiceAttuale].elements+=1;
    }

    unsigned int numeroNz=matricePointer->nz;
    *csrPointer=malloc(sizeof(struct MatriceCsr));
    struct MatriceCsr *matrixCsr=*csrPointer;
    matrixCsr->nz=numeroNz;
    matrixCsr->height=nRighe;
    matrixCsr->width=matricePointer->width;
    matrixCsr->iRP=malloc(sizeof(unsigned int)*(nRighe+1));
    matrixCsr->jValori=malloc(sizeof(unsigned int)*numeroNz);
    matrixCsr->valori=malloc(sizeof(double)*numeroNz);
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
}