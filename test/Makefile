# all: main.cpp
# 	g++ main.cpp ../NeuralNetwork/NeuralNetwork.cpp -o main -O3 -g
# 	g++ digitrecognition.cpp ../NeuralNetwork/NeuralNetwork.cpp -o digit -O3

CC 		:= g++
CINCL 	:= -I../
CFLAGS 	:= -O3 -Wall -DDEBUG

.PHONY: clean all 

all: main digit

main: main.o NeuralNetwork.o
	@echo "serving $@"
	@$(CC) $^ -o $@ $(CFLAGS)

digit: digitrecognition.o NeuralNetwork.o
	@echo "serving $@"
	@$(CC) $^ -o $@ $(CFLAGS)

%.o: %.cpp
	@echo "cooking $@"
	@$(CC) $^ -o $@ $(CFLAGS) $(CINCL) -c

NeuralNetwork.o: ../NeuralNetwork/NeuralNetwork.cpp
	@echo "cooking $@"
	@$(CC) $^ -o $@ $(CFLAGS) $(CINCL) -c

clean:
	@rm *.o

