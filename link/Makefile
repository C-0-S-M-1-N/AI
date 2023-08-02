SRC 	:= ../NeuralNetwork/NeuralNetwork.cpp
CARGS 	:= -O3


ifeq (,$(shell which g++))


ifeq (,$(shell which clang++))
$(error "please install g++ or clang++ in order to compile the library")	
else 

CC := clang++

endif

else

CC := g++

endif


.PHONY: build install

all: build 

build: NeuralNetwork.so.1

install: build
ifneq ($(shell id -u), 0)
	@echo "Run make install as root"
	@exit 1
else
	@cp ./NeuralNetwork.so.1 /lib/
endif
uninstall: 
	@rm /lib/NeuralNetwork.so.*

NeuralNetwork.so.1: NeuralNetwork.o
	@echo "making the shadred object . . ."
	@$(CC) $^ -o $@ -shared 

NeuralNetwork.o: $(SRC)
	@echo "building objs . . ."
	@$(CC) $^ -o $@ -fpic $(CARGS) -c
