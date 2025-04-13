#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "matriciOpp.h"
#include "cuda_alex.h"

#include <cstdint>


#include <cuda_runtime.h>
#include <stdio.h>


int main(int argc, char *argv[]) {
    if (argc < 2)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1);
	}
	MatriceRaw *mat;
    int err=loadMatRaw(argv[1],&mat);
    if(err!=1){
        printf("Errore leggendo la matrice");
        return ;
    }
    MatriceCsr *csrMatrice;
    convertRawToCsr(mat,&csrMatrice);
    Vector *vector;
    Vector *result;
    int seed=13;
    int numOfElements=mat->height;
    generate_random_vector(seed, numOfElements, &vector) ;
    generateEmpty(numOfElements,&result);
    
    fprintf(stdout, "nz=%d height=%d width=%d\n", mat->nz, mat->height, mat->width);
    double time;
    multCudaCSRKernelWarp(csrMatrice,vector,result,&time,256);
    printf("GFLOPS %f\n",mat->nz*2/(time*1e9));
    //testVectors(100);
    multCudaCSRKernelLinear(csrMatrice,vector,result,&time,256);
    printf("GFLOPS %f\n",mat->nz*2/(time*1e9));
   }




