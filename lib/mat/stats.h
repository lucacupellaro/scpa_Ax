#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

typedef struct CsvEntry
{
    char *matrixName;
    char *matrixFormat;
    char *mode;
    int numberOfThreads;
    int extra;
    int numberOfMeasures;
    double *measure;
} CsvEntry;

void initializeCsvEntry(struct CsvEntry *entry, const char *matrixName, const char *matrixFormat, const char *mode,
                        int numberOfThreads, int extra, int numberOfMeasures)
{
    entry->matrixName = strdup(matrixName);
    entry->matrixFormat = strdup(matrixFormat);
    entry->mode = strdup(mode);
    entry->numberOfThreads = numberOfThreads;
    entry->extra = extra;
    entry->numberOfMeasures = numberOfMeasures;

    if (!entry->matrixName || !entry->matrixFormat || !entry->mode) {
        fprintf(stderr, "Memory allocation failed during strdup.\n");
        free(entry->matrixName);
        free(entry->matrixFormat);
        free(entry->mode);
        exit(EXIT_FAILURE);
    }

    entry->measure = (double *)malloc(numberOfMeasures * sizeof(double));
    if (entry->measure == NULL)
    {
        fprintf(stderr, "Memory allocation failed for measures.\n");
        free(entry->matrixName);
        free(entry->matrixFormat);
        free(entry->mode);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numberOfMeasures; i++)
    {
        entry->measure[i] = 0.0;
    }
}

void freeCsvEntry(struct CsvEntry *entry)
{
    if (entry == NULL) return;
    free(entry->matrixName);
    free(entry->matrixFormat);
    free(entry->mode);
    free(entry->measure);
}

FILE *initialize_csv_file(const char *filename)
{
    FILE *file = fopen(filename, "w+");
    if (file == NULL)
    {
        perror("Error opening file for writing");
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return NULL;
    }

    int header_written = fprintf(file, "Matrix Name,Matrix Format,Mode,Number of Threads,Extra,Measure Index,Measure Value\n");

    if (header_written < 0) {
        perror("Error writing header to file");
        fprintf(stderr, "Failed to write header to file: %s\n", filename);
        fclose(file);
        return NULL;
    }

    if (ferror(file)) {
         fprintf(stderr, "Stream error after writing header to file: %s\n", filename);
         fclose(file);
         return NULL;
    }

    return file;
}

void append_csv_entry(FILE *file, const struct CsvEntry *entry)
{
    if (file == NULL)
    {
        fprintf(stderr, "Error: Cannot append to a NULL file pointer.\n");
        return;
    }
     if (entry == NULL)
    {
        fprintf(stderr, "Error: Cannot append data from a NULL CsvEntry pointer.\n");
        return;
    }

    for (int j = 0; j < entry->numberOfMeasures; j++)
    {
        int written = fprintf(file, "%s,%s,%s,%d,%d,%d,%.4f\n",
                              entry->matrixName ? entry->matrixName : "NULL",
                              entry->matrixFormat ? entry->matrixFormat : "NULL",
                              entry->mode ? entry->mode : "NULL",
                              entry->numberOfThreads,
                              entry->extra,
                              j,
                              entry->measure[j]
                              );

        if (written < 0) {
             perror("Error writing entry data to file");
             fprintf(stderr, "Failed to write data for matrix: %s\n", entry->matrixName ? entry->matrixName : "UNKNOWN");
        }
    }
}
