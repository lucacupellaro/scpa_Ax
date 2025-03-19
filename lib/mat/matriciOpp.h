typedef struct MatriceRaw{ // Importante: Le matrici potrebbero essere state salvate sia per righe che per colonna, nel nostro caso quasi sempre per colonne
    unsigned int width,height;
    unsigned int nz;
    unsigned int *iVettore;
    unsigned int *jVettore;
    double *valori;
};

int loadMatRaw(char *filePath, struct MatriceRaw ** matricePointer);
int freeMatRaw(struct MatriceRaw ** matricePointer);


typedef struct MatriceCsr{
    unsigned int width,height;
    unsigned int nz; 
    unsigned int *iRP;   
    unsigned int *jValori;
    double *valori;
 };

 int convertRawToCsr(struct MatriceRaw * matricePointer,struct MatriceCsr **csrPointer);
 int freeMatCsr(struct MatriceCsr ** matricePointer);
