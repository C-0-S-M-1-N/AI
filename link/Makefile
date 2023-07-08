SRC 	:= ../NeuralNetwork/NeuralNetwork.cpp
CC 		:= g++
CARGS 	:= -O3

NeuralNetwork.so: NeuralNetwork.o
	$(CC) $^ -o $@ -shared 

NeuralNetwork.o: $(SRC)
	$(CC) $^ -o $@ -fpic $(CARGS) -c
