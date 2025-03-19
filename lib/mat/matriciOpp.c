#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "matriciOpp.h"

int loadMatRaw(char *filePath, struct MatriceRaw ** matricePointer)
{
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;   

    if ((f = fopen(filePath, "r")) == NULL) {
            printf("Non trovo %s\n",filePath);
            return 0;}
    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return 0;
    }

    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */
    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        return 0;
    }

    /* find out size of sparse matrix .... */
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)return -2;
    /* reseve memory for matrices */

    //Inizializzando la struttura
    *matricePointer=malloc(sizeof(struct MatriceRaw));
    struct MatriceRaw *matrice = *matricePointer;
    matrice->height=M;
    matrice->width=N;
    matrice->nz=nz;
    matrice->iVettore = (int *) malloc(nz * sizeof(int));
    matrice->jVettore = (int *) malloc(nz * sizeof(int));
    matrice->valori = (double *) malloc(nz * sizeof(double));


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (int i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &matrice->iVettore[i], &matrice->jVettore[i], &matrice->valori[i]);
        matrice->iVettore[i]--;  /* adjust from 1-based to 0-based */
        matrice->jVettore[i]--;
    }

    if (f !=stdin) fclose(f);

    /************************/
    /* now write out matrix */
    /************************/

    #ifdef DEBUG 
    printf("ciao")
    mm_write_banner(stdout, matcode);
    mm_write_mtx_crd_size(stdout, M, N, nz);
    for (i=0; i<nz; i++)
        fprintf(stdout, "%d %d %20.19g\n", I[i]+1, J[i]+1, val[i]);
    #endif
    

    
	return 1;
}


int freeMatRaw(struct MatriceRaw ** matricePointer){
    struct MatriceRaw *mat=*matricePointer;
    free(mat->iVettore);
    free(mat->jVettore);
    free(mat->valori);
    free(matricePointer);
    return 0;
}
