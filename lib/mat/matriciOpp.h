typedef struct MatriceRaw{ // Importante: Le matrici potrebbero essere state salvate sia per righe che per colonna, nel nostro caso quasi sempre per colonne
    int width,height;
    int nz;
    int *iVettore;
    int *jVettore;
    double *valori;
};

int loadMatRaw(char *filePath, struct MatriceRaw ** matricePointer);
int freeMatRaw(struct MatriceRaw ** matricePointer);


typedef struct MatriceCsr{
    int width,height;
    int nz; 
    int *iRP;   
    int *jValori;
    double *valori;
 };

 int convertRawToCsr(struct MatriceRaw * matricePointer);