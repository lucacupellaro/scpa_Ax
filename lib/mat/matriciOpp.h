typedef struct MatriceRaw{
    int width,height;
    int nz;
    int *iVettore;
    int *jVettore;
    double *valori;
};