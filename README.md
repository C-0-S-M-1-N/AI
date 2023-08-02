
# Usage  
first off start by including the library with `#include <NeuralNetwork/NeuralNetwork.hpp>`
`AI::NeuralNetwork` provides tow constructors, one for an precomputed weight map and one for creating an neural network from scratch  

## With the source
make sure to compile `NeuralNetwork.cpp` and link the object with the project

## With the shared library

### Linux
! make sure you [installed](##Linux) the library  
- at linking time include `-lNeuralNetwork`  

### Windows
! make sure you [compiled](##Windows) the library  
- copy the dll file to the binary location 
- at linking time include `-lNeuralNetwork`

# Documentation  
- `AI::NeuralNetwork::NeuralNetwork(const std::vector<int>& blueprint, std::function<long double(long double)> f, df, long doube alpha, eta)`  
    - takes the blueprint of the neural net, `blueprint` gives the class the requiered information about how many neurons are per layer and the number of layers (given by blueprint.size())
        ```C++
        [...]
        std::vector<int> blueprint = {2, 10, 15, 4};
        AI::NeuralNetwork nn(blueprint);
        ```  
        the example above creates a NN that has 2 neurons in the input layer, 2 hidden layers, one having 10 neurons and the second 15, and the output layer having 4 neurons.  
    - generates the weights based on a clever tehnique called Xavier's algorithm  
    - takes the activation function and its derivative, `f` and `df` -- default is the linear function  
    - takes tow argumenst alpha, the momentum for backpropagation, and eta, the learing rate -- default is 0.8 and 0.0001

- `AI::NeuralNetwork::FeedInData(const std::vector<double>& data)`  
    - take the input data in a form of an array, `data` must have the same size as `blueprint[0]`  

- `AI::NeuralNetwork::getData(std::vector<double>& data)`
    - calculates the output of the output of the NN  
        ! `AI::NeuralNetwork::FeedInData(...)` calculates the output, `AI::NeuralNetwork::getData(...)` only copies the output of it into `data`  
- `AI::NeuralNetwork::Backpropagation(const std::vector<double>& outputData)`
    - applies backpropagation over the NN to train it, it take as a parameter the correct output of the training set and retweakes the weights accordingly to minimise the error
- `AI::NeuralNetwork::Backpropagation(const std::string& file, ...)`
    - provinding the filepath to the already computed weight, is loads them into the neural net
    - takes the activation function and its derivatie -- default is the linear function  
    - it remembers the learing rate and momentum, its not requiered to pass as an argument  

## examples  
find them in the `./test` directory  

## more about activation function  
the library dedicates a namespace inside `AI::` for some popular activation functions, and their derivatives  
some of them are: sigmoid, hiperbolic tangent, rectifed linear unit (ReLU),
parameter rectified linear unit (PReLU) (and its parameter, as default beeing 0.01),
linear, binary step

about PReLU, the parameter can be modified by modifing the `AI::functions::PReLU_argument` variable

# Compiling the program from source

## Linux
requiered: a c++ compiler
optionaly: gnu make

### Using the Makefile
head to `./link/`: 
- building:     `make build`
- installing:   `make install`
- uninstalling  `make uninstall`

### Using bash commands
head to the root directory (`.`)
- building:  
```bash
g++ ./NeuralNetwork/NeuralNetwork.cpp -o NeuralNetwork.o -fpic -O3 -Wall -c \
g++ NeuralNetwork.o -o NeuralNetwork.so.1 -shared \
```  
- installing:
`sudo cp NeuralNetwork.so.1 /lib/`  
- uninstalling:
`sudo rm /lib/NeuralNetwork.so.*`

## Windows
requiered: a c++ compiler (preferably g++ or clang)

happy coding !
