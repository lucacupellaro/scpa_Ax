
#define RANDOM_VECTOR_MAX_VALUE 1.0

typedef struct Vector{
    unsigned int righr;
    double *vettore;
};
int generate_random_vector(int seed, unsigned int righe, struct Vector **pointerToVector);
int freeRandom(struct Vector **pointerToVector);
int generateEmpty(unsigned int rows,struct Vector **vettore);
void printVector(struct Vector *vec);





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
    unsigned int *iRP;   // array of height+1 lenght
    unsigned int *jValori; // array of size nz
    double *valori;    // array of size
 };

int serialCsrMult(struct MatriceCsr *csr,struct Vector *vec,struct Vector *result);
int serialCsrMultWithTime(struct MatriceCsr *csr,struct Vector *vec,struct Vector *result,double *execTime);
int convertRawToCsr(struct MatriceRaw * matricePointer,struct MatriceCsr **csrPointer);
int freeMatCsr(struct MatriceCsr ** matricePointer);


 