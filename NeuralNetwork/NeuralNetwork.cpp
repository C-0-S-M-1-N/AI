#include <cstdlib>
#include <random>
#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <functional>
#include "NeuralNetwork.hpp"

#define euler 2.71828

#ifdef DEBUG
#include <chrono>

void debug(const char* msj){
	std::cerr << "DEBUG: " << msj << '\n';
}

void debug(const std::string& msj){
	std::cerr << "DEBUG: " << msj << '\n';
}

#else
void debug(const char* msj){ return; } //it gets optimized away
void debug(const std::string& msj){ return; }
#endif

std::function<long double(long double)> f, df;

AI::Functions::functions AI::Functions::sigmoid = {
	[](long double x) -> long double{ return 1.0/(1.0 + 1.0/std::pow(euler, x)); },
	[](long double x) -> long double{
		long double fast = 1.0/(1.0 + 1.0/std::pow(euler, x));
		return fast*(1 - fast);
	}
};

AI::Functions::functions AI::Functions::tanh = {
	[](long double x) -> long double{ return std::tanh(x); },
	[](long double x) -> long double{ return (1.0 - std::tanh(x)*std::tanh(x)); }
};

AI::Functions::functions AI::Functions::ReLU = {
	[](long double x) -> long double{ return x > 0 ? x : 0; },
	[](long double x) -> long double{ return x > 0 ? 1 : 0; }
};

AI::Functions::functions AI::Functions::linear = {
	[](long double x) -> long double{ return x; },
	[](long double x) -> long double{ return 1; }
};

long double AI::Functions::PReLU_argument = 0.01;

AI::Functions::functions AI::Functions::PReLU = {

	[](long double x) -> long double{ return x > 0 ? PReLU_argument*x : x; },
	[](long double x) -> long double{ return x > 0 ? PReLU_argument : 1; }
};

AI::Functions::functions AI::Functions::binaryStep = {
	[](long double x) -> long double{ return x >= 0 ? 1 : 0; },
	[](long double x) -> long double{ return 0; }
};

long double AI::alpha;
long double AI::eta;

/***************NEURAL NETWORK***************/
AI::NeuralNetwork::NeuralNetwork(const std::vector<int>& topology,
				   				 std::function<long double(long double)> activation,
								 std::function<long double(long double)> derivative,
								 long double alpha, long double eta){
	for(size_t i = 0; i < topology.size(); i++){
		net.push_back(std::vector<Neuron>());
		for(int j = 0; j <= topology[i]; j++){
			int fwd = i >= topology.size() - 1 ? 0 : topology[i+1];

			net.back().push_back(Neuron(fwd, j));
		}
	}
	for(size_t i = 0; i < topology.size(); i++){
		net[i].back().OutputVal = 0;
	}

	f  = activation;
	df = derivative;
	AI::alpha = alpha;
	AI::eta = eta;
}

void AI::NeuralNetwork::FeedInData(const std::vector<long double>& data){
	assert(data.size() == net[0].size()-1); //error checking
	
	for(size_t i = 0; i < data.size(); i++){
		net[0][i].OutputVal = data[i];
	}

	//do math!
	for(size_t i = 1; i < net.size(); i++){
		for(size_t j = 0; j < net[i].size(); j++){
			net[i][j].activate(net[i-1]);
		}
	}

}


void AI::NeuralNetwork::Backpropagation(const std::vector<long double>& data){
	
	assert(data.size() == net.back().size() - 1);

	//calculate errors and gradients

	for(size_t i = 0; i < net.back().size() - 1; i++){
		net.back()[i].calculateOutputGradient(data[i]);
	}


	for(size_t i = net.size() - 2; i > 0; i--){
		for(size_t j = 0; j < net[i].size() - 1; j++){
			net[i][j].calculateHiddenLayerGradiend(net[i+1]);
		}
	}
	//adjusting weights
	for(size_t i = net.size() - 1; i > 0; i--)	{
		for(size_t j = 0; j < net[i].size() - 1; j++){
			net[i][j].updateWeights(net[i-1]);
		}
	}
}

void AI::NeuralNetwork::getData(std::vector<long double>& data) const {
	data.clear();
	for(size_t i = 0; i < net.back().size() - 1; i++){
		data.push_back(net.back()[i].OutputVal);
	}
}

void AI::NeuralNetwork::NNexportHelper(std::fstream& output) const{
	size_t consts_to_write;
	long double weight;
	for(size_t i = 0; i < net.size(); i++){
		consts_to_write = net[i].size(); // number of neurons in the layer i
		output.write(reinterpret_cast<char*>(&consts_to_write), sizeof(consts_to_write));
		debug(std::to_string(consts_to_write));

		for(size_t j = 0; j < net[i].size(); j++){
			std::vector<connections> neuronWeights = net[i][j].getConnections();
			consts_to_write = neuronWeights.size(); // number of neurons
													// in the nex layer
													// the same as the number 
													// of weights

			output.write(reinterpret_cast<char*>(&consts_to_write), sizeof(consts_to_write));
			for(size_t k = 0; k < neuronWeights.size(); k++){
				weight = neuronWeights[k].weight; // the weight

				output.write(reinterpret_cast<char*>(&weight), sizeof(weight));
			}

		}

	}

}

void AI::NeuralNetwork::exportData(const std::string& outFile) const{
	std::fstream output;
	output.exceptions(std::ios::badbit | std::ios::failbit);
	try{
		output.open(outFile, std::ios::out | std::ios::binary);
		
		size_t consts_to_write;
		
		consts_to_write = net.size();
		output.write(reinterpret_cast<char*>(&consts_to_write), sizeof(consts_to_write));


		debug(std::to_string(consts_to_write));
		NNexportHelper(output);	

	}
	catch(const std::fstream::failure& e){
		std::cerr << "can't create/open the file\n" << e.what() << std::endl;
	}
	output.write(reinterpret_cast<char*>(&alpha), sizeof(alpha));
	output.write(reinterpret_cast<char*>(&eta), sizeof(eta));
}

AI::NeuralNetwork::NeuralNetwork(const std::string& file,
								 std::function<long double(long double)> activation,
								 std::function<long double(long double)> derivative){
	std::fstream input;
// 	input.exceptions(std::ios::badbit | std::ios::failbit);
	
	debug(file.c_str());

	try{
		input.open(file, std::ios::in | std::ios::binary);

		long double weight;
		size_t layers;
		input.read(reinterpret_cast<char*>(&layers), sizeof(layers));
		
		debug(std::to_string(layers).c_str());	

		for(size_t i = 0; i < layers; i++){

			net.push_back(std::vector<Neuron>());
			size_t neurons; 
			input.read(reinterpret_cast<char*>(&neurons), sizeof(neurons));
			debug(std::to_string(neurons));

			for(size_t j = 0; j < neurons; j++){ 

				size_t next_layer_neurons; 
				input.read(reinterpret_cast<char*>(&next_layer_neurons), 
						   sizeof(next_layer_neurons));
// 				debug(std::to_string(next_layer_neurons));
				std::vector<long double> weights;

				for(size_t k = 0; k < next_layer_neurons; k++){ 
					input.read(reinterpret_cast<char*>(&weight), sizeof(weight));
					weights.push_back(weight);
				}
				net[i].push_back(Neuron(next_layer_neurons, j, 0, &weights));
// 				debug("EOLAYER");
			}
		}

	} catch (const std::ios::failure& e){
		std::cerr << "choudn't open the file " << file << '\n'
				<< e.what() << std::endl;
	}
	f = activation;
	df = derivative;
	

	input.read(reinterpret_cast<char*>(&alpha), sizeof(alpha));
	input.read(reinterpret_cast<char*>(&eta), sizeof(eta));
}

/************NEURON****************/


AI::Neuron::Neuron(size_t fwdElements, size_t i, bool randW,
				const std::vector<long double>* v){
	ith = i;
	if(randW){
		std::random_device rd;
		std::mt19937 generator(rd());

		long double xavierFactor = 1.0/sqrt(fwdElements);
		
		for(size_t i = 0; i < fwdElements; i++){
			std::normal_distribution<double> dist(0, xavierFactor);
			long double w = dist(generator);
			fwd.push_back({w, 0});
		}
	}
	else
		for(size_t i = 0; i < fwdElements; i++){
			fwd.push_back({(*v)[i], 0});
		}
}

void AI::Neuron::updateWeights(std::vector<Neuron>& lastLayer){
	for(size_t i = 0; i < lastLayer.size() - 1; i++){
		long double delta = 
				AI::eta * lastLayer[i].OutputVal * Gradient
				+ AI::alpha * lastLayer[i].fwd[ith].deltaweight
				;
		lastLayer[i].fwd[ith].deltaweight = delta;
		lastLayer[i].fwd[ith].weight += delta;
// 		Gradient = 0;
	}
}

void AI::Neuron::activate(const std::vector<Neuron>& lastLayer){
	OutputVal = 0;
	long double totalX = 0;
	long double bais = lastLayer.back().OutputVal;
	for(size_t i = 0; i < lastLayer.size() - 1; i++){
		totalX += lastLayer[i].OutputVal*lastLayer[i].fwd[ith].weight;
	}
	totalX += bais;
	OutputVal = f(totalX);
// 	if(OutputVal == 0)
// 		std::cerr << totalX << '\n';
}

void AI::Neuron::calculateOutputGradient(long double data){
	Gradient = (data - OutputVal) * df(OutputVal);
}

void AI::Neuron::calculateHiddenLayerGradiend(const std::vector<AI::Neuron>& nextLayer){
	long double sum = 0;
	for(size_t i = 0; i < nextLayer.size() - 1; i++){
		sum += fwd[i].weight * nextLayer[i].Gradient;
	}

	Gradient = sum * df(OutputVal);
}
