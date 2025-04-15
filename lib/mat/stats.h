#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct CsvEntry
{
    char *matrixName;
    char *matrixFormat;
    char *mode;
    char *configuration;
    int nz;
    int hack;
    int numberOfThreads;
    int threadsForBlock;
    int numberOfMeasures;
    double *measure;
    double errorValue;
    double errorPercentage;
} CsvEntry;

void initializeCsvEntry(CsvEntry *entry,
                        const char *matrixName,
                        const char *matrixFormat,
                        const char *mode,
                        const char *configuration,
                        int nz,
                        int hack,
                        int numberOfThreads,
                        int threadsForBlock,
                        int numberOfMeasures,
                        double errorValue,
                        double errorPercentage)
{
    entry->matrixName = strdup(matrixName);
    entry->matrixFormat = strdup(matrixFormat);
    entry->mode = strdup(mode);
    entry->configuration = strdup(configuration);

    if (!entry->matrixName || !entry->matrixFormat || !entry->mode || !entry->configuration)
    {
        fprintf(stderr, "Memory allocation failed during strdup.\n");
        free(entry->matrixName);
        free(entry->matrixFormat);
        free(entry->mode);
        free(entry->configuration);
        exit(EXIT_FAILURE);
    }

    entry->nz = nz;
    entry->hack = hack;
    entry->numberOfThreads = numberOfThreads;
    entry->threadsForBlock = threadsForBlock;
    entry->numberOfMeasures = numberOfMeasures;
    entry->errorValue = errorValue;
    entry->errorPercentage = errorPercentage;

    entry->measure = NULL;
    if (numberOfMeasures > 0)
    {
        entry->measure = (double *)malloc(numberOfMeasures * sizeof(double));
        if (entry->measure == NULL)
        {
            fprintf(stderr, "Memory allocation failed for measures array.\n");
            free(entry->matrixName);
            free(entry->matrixFormat);
            free(entry->mode);
            free(entry->configuration);
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < numberOfMeasures; i++)
        {
            entry->measure[i] = 0.0;
        }
    }
}

void freeCsvEntry(CsvEntry *entry)
{
    if (entry)
    {
        free(entry->matrixName);
        free(entry->matrixFormat);
        free(entry->mode);
        free(entry->configuration);
        free(entry->measure);
        entry->matrixName = NULL;
        entry->matrixFormat = NULL;
        entry->mode = NULL;
        entry->configuration = NULL;
        entry->measure = NULL;
    }
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

    int header_written = fprintf(file, "Matrix Name,Matrix Format,Mode,Configuration,nz,Hack,Threads,Threads for block,Measure Index,Measure Value,error,error percentage\n");

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

void append_csv_entry(FILE *file, const CsvEntry *entry)
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
    // Add a check for the measure array itself if measures are expected
    if (entry->measure == NULL && entry->numberOfMeasures > 0)
    {
         fprintf(stderr, "Error: CsvEntry measure pointer is NULL but numberOfMeasures is %d.\n", entry->numberOfMeasures);
         return;
    }


    for (int j = 0; j < entry->numberOfMeasures; j++)
    {
        // Match the fprintf format to the header:
        // "Matrix Name,Matrix Format,Mode,Configuration,nz,Hack,Threads,Threads for block,Measure Index,Measure Value,error,error perc\n"
        int written = fprintf(file, "%s,%s,%s,%s,%d,%d,%d,%d,%d,%.8g,%.8g,%.4f\n",
                              entry->matrixName ? entry->matrixName : "NULL",
                              entry->matrixFormat ? entry->matrixFormat : "NULL",
                              entry->mode ? entry->mode : "NULL",
                              entry->configuration ? entry->configuration : "NULL",
                              entry->nz,
                              entry->hack,
                              entry->numberOfThreads,
                              entry->threadsForBlock, // Use the correct field name
                              j,                      // Measure Index
                              entry->measure[j],      // Measure Value
                              entry->errorValue,
                              entry->errorPercentage
                              );

        if (written < 0) {
             perror("Error writing entry data to file");
             // Provide more context in the error message if possible
             fprintf(stderr, "Failed to write data row index %d for matrix: %s\n", j, entry->matrixName ? entry->matrixName : "UNKNOWN");
             // Consider if you should stop processing this entry or the entire file
             // return; // Uncomment to stop after the first write error for this entry
        }
    }

    // Flushing after the loop might be more efficient than flushing every line
    fflush(file);
}