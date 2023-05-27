all: main.cpp
	g++ main.cpp ./NeuralNetwork/NeuralNetwork.cpp -o main -O3
