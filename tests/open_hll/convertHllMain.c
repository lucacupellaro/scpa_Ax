
#include <stdio.h>
#include <stdlib.h>
#include "matriciOpp.h"
int main(int argc, char *argv[])
{

    struct MatriceHLL *matHll;
    struct MatriceRaw *mat;
   

    if (argc < 3)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1);
	}
    
    int result=loadMatRaw(argv[1],&mat);
    if(result!=1){
        printf("Errore leggendo la matrice");
        return 0;
    }

    fprintf(stdout, "nz=%d height=%d width=%d\n", mat->nz, mat->height, mat->width); 
    for(int i=0;i<mat->nz;i++){
        fprintf(stdout, "%d %d %20.19g\n", mat->iVettore[i]+1, mat->jVettore[i]+1, mat->valori[i]);
    }
    
    int hack = atoi(argv[2]);
    printf("%d",hack);


    
    int result2=convertRawToHll(mat,hack,&matHll);
    if(result2 != 1){
        printf("Error building HLL matrix, error code: %d\n", result2);
        return result2; // restituisce il codice d'errore ricevuto
    }

    
   
    result2=printHLL(&matHll);
    if(result2!=1){
        printf("Error while reading HLL matrix");
        return 0;
    }
    
    
	return 0;
}

