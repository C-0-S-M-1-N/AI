all: main.cpp
	g++ main.cpp ./NeuralNetwork/NeuralNetwork.cpp -o main -O3 -g
	g++ digitrecognition.cpp NeuralNetwork/NeuralNetwork.cpp -o digit -O3

