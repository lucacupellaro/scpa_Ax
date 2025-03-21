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

 int convertRawToCsr(struct MatriceRaw * matricePointer,struct MatriceCsr **csrPointer);
 int freeMatCsr(struct MatriceCsr ** matricePointer);

// Struttura del blocco in formato ELLPACK
typedef struct {
    int M;      // Righe del blocco (HackSize o meno)
    int N;      // Colonne del blocco
    int MAXNZ;  // Max non-zero per riga
    int** JA;   // Indici di colonna
    double** AS;// Valori dei coefficienti
} ELLPACK_Block;

// Struttura della matrice HLL
typedef struct MatriceHLL {
    int totalRows;        // Numero totale di righe della matrice globale
    int totalCols;        // Numero totale di colonne della matrice globale
    int HackSize;         // Numero di righe per blocco
    int numBlocks;        // Numero di blocchi totali
    ELLPACK_Block** blocks; // Array di blocchi
} MatriceHLL;



int convertRawToEllpack(struct MatriceRaw* matricePointer, int acksize, ELLPACK_Block** block);

int convertRawToHll(struct MatriceRaw *matricePointer, int hackSizeP, struct MatriceHLL **hll);
