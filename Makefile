.PHONY: run-test-cmake clean

# Directory for building

BUILDS_FOLDER=builds/
TEST_FOLDER=tests/

TEST_FOLDER_CMAKE=$(TEST_FOLDER)cmake
BUILD_DIR_TEST_CMAKE = $(BUILDS_FOLDER)$(TEST_FOLDER_CMAKE)

TEST_FOLDER_MATRICI=$(TEST_FOLDER)open_matrici
BUILD_DIR_TEST_MATRICI=$(BUILDS_FOLDER)$(TEST_FOLDER_MATRICI)

TEST_OPEN_HLL=$(TEST_FOLDER)open_hll
BUILD_DIR_OPEN_HLL=$(BUILDS_FOLDER)$(TEST_OPEN_HLL)

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

build-test-HLL:
	mkdir -p $(BUILD_DIR_OPEN_HLL)
	cd $(BUILD_DIR_OPEN_HLL) && cmake $(CURRENT_DIR)/$(TEST_OPEN_HLL)
	cd $(BUILD_DIR_OPEN_HLL) && cmake --build .
	#cd $(BUILD_DIR_OPEN_HLL) && ./Main

run-test-matrici:
	echo "Checking parameters..."
	if [ -z "$(MATRICE)" ]; then \
        echo "ERROR: MATRICE PATH is not set! put MATRICE=PATH at the end"; \
        exit 1; \
    fi
	cd $(BUILD_DIR_TEST_MATRICI) && ./Main $(CURRENT_DIR)/$(MATRICE)

run-test-HLL:
	echo "Checking parameters..."
	if [ -z "$(MATRICE)" ]; then \
        echo "ERROR: MATRICE PATH is not set! put MATRICE=PATH at the end"; \
        exit 1; \
    fi 
	cd $(BUILD_DIR_OPEN_HLL) && ./Main $(CURRENT_DIR)/$(MATRICE) $(P)
