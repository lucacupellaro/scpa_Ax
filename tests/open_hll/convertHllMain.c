
#include <stdio.h>
#include <stdlib.h>
#include "matriciOpp.h"
int main(int argc, char *argv[])
{

    if (argc < 3)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1);
	}
    struct MatriceRaw *mat;
    int result=loadMatRaw(argv[1],&mat);
    if(result!=1){
        printf("Errore leggendo la matrice");
        return 0;
    }
    fprintf(stdout, "nz=%d height=%d width=%dn", mat->nz, mat->height, mat->width);
    for(int i=0;i<mat->nz;i++){
        fprintf(stdout, "%d %d %20.19g\n", mat->iVettore[i]+1, mat->jVettore[i]+1, mat->valori[i]);
    }
    struct MatriceHLL *matHll;
    int hack = atoi(argv[2]);
    int result2=convertRawToHll(&mat,hack,&matHll);
    if(result2!=1){
        printf("Errore leggendo la matrice");
        return 0;
    }
	return 0;
}

