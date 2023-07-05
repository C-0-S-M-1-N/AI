# This is basic Neural Network written in C++  

## Usage  
first off start by including the library with `#include <NeuralNetwork/NeuralNetwork.cpp>`
the class is in the `AI` namespace, `AI::NeuralNetwork` provides tow constructors, one for an precomputed weight map and one for creating an neural network from scratch  
    
## Documentation  
- `AI::NeuralNetwork::NeuralNetwork(const std::vector<int>& blueprint)`  
    - takes the blueprint of the neural net, `blueprint` gives the class the requiered information about how many neurons are per layer and the number of layers (given by blueprint.size())
        ```C++
        [...]
        std::vector<int> blueprint = {2, 10, 15, 4};
        AI::NeuralNetwork nn(blueprint);
        ```  
        the example above gives an NN that has 2 neurons on input layer, 2 hidden layers, one having 10 neurons and the second 15, and the output layer having 4 neurons.  
    - generates the weights based on a clever tehnique called Xavier's algorithm  
    - TODO: add the flexibility to add custom initializations for weights and custom activation function  

- `AI::NeuralNetwork::FeedInData(const std::vector<double>& data)`  
    - take the input data in a form of an array, `data` must have the same size as `blueprint[0]`  

- `AI::NeuralNetwork::getData(std::vector<double>& data)`
    - calculates the output of the output of the NN  
        !! `AI::NeuralNetwork::FeedInData(...)` calculates the output, `AI::NeuralNetwork::getData(...)` only copies the output of it into `data`  
- `AI::NeuralNetwork::Backpropagation(const std::vector<double>& outputData)`
    - applys backpropagation over the NN to train it, it take as a parameter the correct output of the training set and retweakes the weights accordingly to minimise the error


## examples  
find them in the `./test` directory
