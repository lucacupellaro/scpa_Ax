.PHONY: run-test-cmake clean

# Directory for building
BUILD_DIR_TEST_CMAKE = cmake-test

build-run-test-cmake:
	mkdir -p $(BUILD_DIR_TEST_CMAKE)
	cd $(BUILD_DIR_TEST_CMAKE) && cmake ../tests/cmake
	cd $(BUILD_DIR_TEST_CMAKE) && cmake --build .
	cd $(BUILD_DIR_TEST_CMAKE) && ./Main

run-test-cmake:
	cd $(BUILD_DIR_TEST_CMAKE) && ./Main
clean-test-cmake:
	rm -rf $(BUILD_DIR_TEST_CMAKE)