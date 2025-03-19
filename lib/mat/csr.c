
#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "matriciOpp.h"

typedef struct TempStruct{
    unsigned int elements;
    unsigned int bufferSize;
    unsigned int  *indices;
};

int convertRawToCsr(struct MatriceRaw * matricePointer){ //Devo scorrere iVec e trovare tutti gli elementi di tutte le righe
int nRighe=matricePointer->height;

struct TempStruct *indicesArray=malloc(sizeof(struct TempStruct)*nRighe);
for(int i=0;i<nRighe;i++){
    indicesArray->elements=0;
    indicesArray->bufferSize=4;
    indicesArray->indices=malloc(sizeof(unsigned int)*4);
}
for(int i=0;i<matricePointer->nz;i++) {
        int indiceAttuale=matricePointer->iVettore[i];
        if(indicesArray[indiceAttuale].elements==indicesArray[indiceAttuale].bufferSize){
            indicesArray[indiceAttuale].bufferSize=indicesArray[indiceAttuale].bufferSize*2; 
            indicesArray[indiceAttuale].indices=realloc(indicesArray[indiceAttuale].indices,indicesArray[indiceAttuale].bufferSize);
        }
        indicesArray[indiceAttuale].indices[indicesArray[indiceAttuale].elements]=matricePointer->jVettore[i];
}
}