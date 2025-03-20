
#define PRINT 0
#include <stdio.h>
#include <stdlib.h>
#include "matriciOpp.h"
int main(int argc, char *argv[])
{

    if (argc < 2)
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
    
    fprintf(stdout, "nz=%d height=%d width=%d\n", mat->nz, mat->height, mat->width);
    
    struct MatriceCsr *csrMatrice;
    convertRawToCsr(mat,&csrMatrice);
    unsigned int rows=mat->height;
    struct Vector *vectorR;
    int seed = 42;
    
    if (generate_random_vector(seed, rows, &vectorR) == 0) {
       //printVector(vectorR);
        //freeRandom(&vector);
    } else {
        printf("Failed to allocate memory or invalid input.\n");
        return 1;
    }

    struct Vector *resultV;
    generateEmpty(rows,&resultV);
    double time=0;
    serialCsrMultWithTime(csrMatrice,vectorR,resultV,&time);
    printf("Serial calculation for nz:%u,%f time, %f GFLOPS",mat->nz,time,2.0*mat->nz/(time*1000000000));
    //printVector(resultV);
    freeMatRaw(&mat);
    freeMatCsr(&csrMatrice);
}

