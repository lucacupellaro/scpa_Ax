# scpa_Ax Progetto

Il `.gitignore` ignorerà:  
- File delle matrici (`.mtx`)  
- Cartelle `build`  
- File di configurazione di Visual Studio Code (`.vscode/`)  

## Esecuzione del Test CMake  

Per eseguire il primo test ed eseguire il **"test cmake"**, usare il comando seguente nella  cartella root del proggetto:  

```sh
make build-run-test-cmake
```
Per pulire o rieseguire questo determianto test eseguire :
```sh
make run-test-cmake
make clean-test-cmake
```



## Esecuzione del Test Matrici  

Per stampare la prima matrici eseguire questo test su cage4:

```sh
make build-test-matrici run-test-matrici MATRICE=mat/cage4.mtx
```

\#TODO FARE STRUCT PER CONSERVARE LA MATRICE
