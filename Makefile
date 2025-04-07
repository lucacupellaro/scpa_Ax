.PHONY: run-test-cmake clean

# Directory for building

BUILDS_FOLDER=builds/
TEST_FOLDER=tests/

TEST_FOLDER_CMAKE=$(TEST_FOLDER)cmake
BUILD_DIR_TEST_CMAKE = $(BUILDS_FOLDER)$(TEST_FOLDER_CMAKE)

TEST_FOLDER_MATRICI=$(TEST_FOLDER)open_matrici
BUILD_DIR_TEST_MATRICI=$(BUILDS_FOLDER)$(TEST_FOLDER_MATRICI)

TEST_FOLDER_CSR_MULT=$(TEST_FOLDER)csrMultiplication
BUILD_DIR_TEST_CSR_MULT=$(BUILDS_FOLDER)$(TEST_FOLDER_CSR_MULT)

TEST_OPEN_HLL=$(TEST_FOLDER)open_hll
BUILD_DIR_OPEN_HLL=$(BUILDS_FOLDER)$(TEST_OPEN_HLL)

TEST_OPEN_CUDA=$(TEST_FOLDER)cudaTest
BUILD_DIR_OPEN_CUDA=$(BUILDS_FOLDER)$(TEST_OPEN_CUDA)

TEST_OPEN_CUDAHLL=$(TEST_FOLDER)cudaHll
BUILD_DIR_OPEN_CUDAHLL=$(BUILDS_FOLDER)$(TEST_OPEN_CUDAHLL)

TEST_SAVE_STATS=$(TEST_FOLDER)saveStats
BUILD_DIR_STATS_TEST=$(BUILDS_FOLDER)$(TEST_SAVE_STATS)

CURRENT_DIR := $(shell pwd)



clean:
	rm -rf $(BUILDS_FOLDER)




build-test-cmake:
	mkdir -p $(BUILD_DIR_TEST_CMAKE)
	cd $(BUILD_DIR_TEST_CMAKE) && cmake $(CURRENT_DIR)/$(TEST_FOLDER_CMAKE)
	cd $(BUILD_DIR_TEST_CMAKE) && cmake --build .
	#cd $(BUILD_DIR_TEST_CMAKE) && ./Main

run-test-cmake:
	cd $(BUILD_DIR_TEST_CMAKE) && ./Main


build-test-matrici:
	mkdir -p $(BUILD_DIR_TEST_MATRICI)
	cd $(BUILD_DIR_TEST_MATRICI) && cmake $(CURRENT_DIR)/$(TEST_FOLDER_MATRICI)
	cd $(BUILD_DIR_TEST_MATRICI) && cmake --build .
	#cd $(BUILD_DIR_TEST_MATRICI) && ./Main



run-test-matrici:
	echo "Checking parameters..."
	if [ -z "$(MATRICE)" ]; then \
        echo "ERROR: MATRICE PATH is not set! put MATRICE=PATH at the end"; \
        exit 1; \
    fi
	cd $(BUILD_DIR_TEST_MATRICI) &&  ./Main $(CURRENT_DIR)/$(MATRICE)

	


build-test-csrM:
	mkdir -p $(BUILD_DIR_TEST_CSR_MULT)
	cd $(BUILD_DIR_TEST_CSR_MULT) && cmake $(CURRENT_DIR)/$(TEST_FOLDER_CSR_MULT)
	cd $(BUILD_DIR_TEST_CSR_MULT) && cmake --build .
	#cd $(BUILD_DIR_TEST_CSR_MULT) && ./Main

run-test-csrM:
	echo "Checking parameters..."
	if [ -z "$(MATRICE)" ]; then \
        echo "ERROR: MATRICE PATH is not set! put MATRICE=PATH at the end"; \
        exit 1; \
    fi
	cd $(BUILD_DIR_TEST_CSR_MULT) && ./Main $(CURRENT_DIR)/$(MATRICE)

build-test-HLL:
	mkdir -p $(BUILD_DIR_OPEN_HLL)
	cd $(BUILD_DIR_OPEN_HLL) && cmake $(CURRENT_DIR)/$(TEST_OPEN_HLL)
	cd $(BUILD_DIR_OPEN_HLL) && cmake --build .
	#cd $(BUILD_DIR_OPEN_HLL) && ./Main



run-test-HLL:
	echo "Checking parameters..."
	if [ -z "$(MATRICE)" ]; then \
        echo "ERROR: MATRICE PATH is not set! put MATRICE=PATH at the end"; \
        exit 1; \
    fi 
	cd $(BUILD_DIR_OPEN_HLL) && valgrind ./Main $(CURRENT_DIR)/$(MATRICE) $(P)


build-test-stats:
	mkdir -p $(BUILD_DIR_STATS_TEST)
	cd $(BUILD_DIR_STATS_TEST) && cmake $(CURRENT_DIR)/$(TEST_SAVE_STATS)
	cd $(BUILD_DIR_STATS_TEST) && cmake --build .
	#cd $(BUILD_DIR_STATS_TEST) && ./Main



run-test-stats:
	@echo "Checking parameters..."
	@{ \
	    if [ -z "$(MATRICE)" ]; then \
	        echo "ERROR: MATRICE PATH is not set! Use MATRICE=PATH"; exit 1; \
	    fi; \
	    if [ -z "$(THREADS)" ]; then \
	        echo "ERROR: THREADS is not set! Use THREADS=NUM"; exit 1; \
	    fi; \
	    if [ -z "$(ITERATIONS)" ]; then \
	        echo "ERROR: ITERATIONS is not set! Use ITERATIONS=NUM"; exit 1; \
	    fi; \
	    if [ -z "$(HACK)" ]; then \
	        echo "ERROR: HACK is not set! Use HACK=VALUE"; exit 1; \
	    fi; \
	    echo "All parameters are set. Running test..."; \
	    cd $(BUILD_DIR_STATS_TEST) && ./Main $(CURRENT_DIR)/$(MATRICE) $(THREADS) $(ITERATIONS) $(HACK); \
	}


run-test-cuda:
	echo "Checking parameters..."
	if [ -z "$(MATRICE)" ]; then \
        echo "ERROR: MATRICE PATH is not set! put MATRICE=PATH at the end"; \
        exit 1; \
    fi 
	cd $(BUILD_DIR_OPEN_CUDA) && ./Main $(CURRENT_DIR)/$(MATRICE) $(P)


build-test-cuda:
	mkdir -p $(BUILD_DIR_OPEN_CUDA)
	cd $(BUILD_DIR_OPEN_CUDA) && cmake $(CURRENT_DIR)/$(TEST_OPEN_CUDA)
	cd $(BUILD_DIR_OPEN_CUDA) && cmake --build .
	#cd $(BUILD_DIR_OPEN_CUDA) && ./Main


run-test-cudaHll:
	echo "Checking parameters..."
	if [ -z "$(MATRICE)" ]; then \
        echo "ERROR: MATRICE PATH is not set! put MATRICE=PATH at the end"; \
        exit 1; \
    fi 
	cd $(BUILD_DIR_OPEN_CUDAHLL) && ./Main $(CURRENT_DIR)/$(MATRICE) $(P)


build-test-cudaHll:
	mkdir -p $(BUILD_DIR_OPEN_CUDAHLL)
	cd $(BUILD_DIR_OPEN_CUDAHLL) &&  cmake $(CURRENT_DIR)/$(TEST_OPEN_CUDAHLL)
	cd $(BUILD_DIR_OPEN_CUDAHLL) && cmake --build .
	#cd $(BUILD_DIR_OPEN_CUDAHLL) && ./Main

	



