# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -

# Include directories
INCLUDES = -I./

# Source files
SRC = ../weights/weights.cpp ImagePatchEmbedding.cpp 

# Object files (compiled from the source files)
OBJ = $(SRC:.cpp=.o)

# Output executable name
TARGET = imageembedding

# The default target (build the program)
all: $(TARGET)

# Rule to link the object files into the executable
$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $(TARGET)

# Rule to compile the .cpp files into .o object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean up generated files
clean:
	rm -f $(OBJ) $(TARGET)

# Rule for running the executable
run: $(TARGET)
	./$(TARGET)