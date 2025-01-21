# Compiler
NVCC = nvcc

# Compiler flags
CFLAGS = -arch=sm_50 -G -g

# Directories
SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
BIN_DIR = bin

# Source files
SRCS = $(wildcard $(SRC_DIR)/*.cu)

# Object files
OBJS = $(SRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

# Executable
EXEC = $(BIN_DIR)/cuda_project

# Default target
all: $(EXEC)

# Link
$(EXEC): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(CFLAGS) -o $@ $^

# Compile
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(CFLAGS) -I$(INC_DIR) -o $@ -c $<

# Clean
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Phony targets
.PHONY: all clean