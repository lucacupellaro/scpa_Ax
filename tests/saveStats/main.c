#include <omp.h>

#include <stdio.h>
#include <stdlib.h>
#include "matriciOpp.h"
#include "stats.h"
int main(int argc, char *argv[])
{

    if (argc < 5)
    {
        fprintf(stderr, "Usage: %s [martix-market-filename] [number of max threads] [number of measure for combination] [lenght of hll blocks] \n", argv[0]);
        exit(1);
    }
    char *nomeMat = strrchr(argv[1], '/');
    struct MatriceRaw *mat;
    int result = loadMatRaw(argv[1], &mat);
    if (result != 1)
    {
        printf("Errore leggendo la matrice");
        return 0;
    }
    int threads = atoi(argv[2]);
    int iterations = atoi(argv[3]);
    int hack = atoi(argv[4]);
    struct CsvEntry *results = malloc(sizeof(struct CsvEntry) * 4);

    struct MatriceCsr *csrMatrice;
    convertRawToCsr(mat, &csrMatrice);
    unsigned int rows = mat->height;

    struct Vector *vectorR;
    int seed = 42;
    if (generate_random_vector(seed, rows, &vectorR) != 0)
    {
        printf("Failed to allocate memory or invalid input.\n");
        return 1;
    }
    omp_set_num_threads(threads);

    struct Vector *resultV1;
    generateEmpty(rows, &resultV1);
    initializeCsvEntry(&results[0], nomeMat + 1, "csr", "serial", 1, 0, iterations);

    double time = 0;
    for (int i = 0; i < iterations; i++)
    {
        csrMultWithTime(&serialCsrMult, csrMatrice, vectorR, resultV1, &time);
        results[0].measure[i] = 2.0 * mat->nz / (time * 1000000000);
    }
    struct Vector *resultV2;
    generateEmpty(rows, &resultV2);
    initializeCsvEntry(&results[1], nomeMat + 1, "csr", "parallelOpenMp", threads, 0, iterations);

    time = 0;
    for (int i = 0; i < iterations; i++)
    {
        csrMultWithTime(&parallelCsrMult, csrMatrice, vectorR, resultV2, &time);
        results[1].measure[i] = 2.0 * mat->nz / (time * 1000000000);
    }
    freeMatCsr(&csrMatrice);

    struct MatriceHLL *matHll;
    convertRawToHll(mat, hack, &matHll);
    struct Vector *resultV3;
    generateEmpty(rows, &resultV3);
    initializeCsvEntry(&results[2], nomeMat + 1, "hll", "serial", 1, hack, iterations);
    time = 0;
    for (int i = 0; i < iterations; i++)
    {
        hllMultWithTime(&serialMultiplyHLL, matHll, vectorR, resultV3, &time);
        results[2].measure[i] = 2.0 * mat->nz / (time * 1000000000);
    }
    struct Vector *resultV4;
    generateEmpty(rows, &resultV4);
    initializeCsvEntry(&results[3], nomeMat + 1, "hll", "parallelOpenMp", threads, hack, iterations);
    time = 0;
    for (int i = 0; i < iterations; i++)
    {
        hllMultWithTime(&serialMultiplyHLL, matHll, vectorR, resultV3, &time);
        results[3].measure[i] = 2.0 * mat->nz / (time * 1000000000);
    }

    writeCsvEntriesToFile("../../../test.csv", results, 4);
    //    csrMultWithTime(&serialCsrMult,csrMatrice,vectorR,resultV,&time);
    //  printf("Serial calculation for nz:%u,%f time, %f GFLOPS\n",mat->nz,time,2.0*mat->nz/(time*1000000000));
}