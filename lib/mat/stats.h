#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct CsvEntry
{
    char *matrixName;
    char *matrixFormat;
    char *mode;
    int numberOfThreads;
    int extra;
    int numberOfMeasures;
    double *measure;
};

void initializeCsvEntry(struct CsvEntry *entry, const char *matrixName, const char *matrixFormat, const char *mode,
                        int numberOfThreads, int extra, int numberOfMeasures)
{
    entry->matrixName = strdup(matrixName);
    entry->matrixFormat = strdup(matrixFormat);
    entry->mode = strdup(mode);
    entry->numberOfThreads = numberOfThreads;
    entry->extra = extra;
    entry->numberOfMeasures = numberOfMeasures;

    entry->measure = (double *)malloc(numberOfMeasures * sizeof(double));
    if (entry->measure == NULL)
    {
        fprintf(stderr, "Memory allocation failed for measures.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numberOfMeasures; i++)
    {
        entry->measure[i] = 0.0;
    }
}

void writeCsvEntriesToFile(const char *filename, struct CsvEntry *entries, int n)
{
    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        fprintf(stderr, "Failed to open file for writing.\n");
        return;
    }

    // Write header
    fprintf(file, "Matrix Name,Matrix Format,Mode,Number of Threads,Extra,Measure Index,Measure Value\n");

    // Write entries
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < entries[i].numberOfMeasures; j++)
        {
            fprintf(file, "%s,%s,%s,%d,%d,%d,%.4f\n", entries[i].matrixName, entries[i].matrixFormat, entries[i].mode,
                    entries[i].numberOfThreads, entries[i].extra, j, entries[i].measure[j]);
        }
    }

    fclose(file);
}

void freeCsvEntry(struct CsvEntry *entry)
{
    free(entry->matrixName);
    free(entry->matrixFormat);
    free(entry->mode);
    free(entry->measure);
    free(entry);
}