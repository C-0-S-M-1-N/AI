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
	for(int i = 0; i < topology.size(); i++){
		net.push_back(std::vector<Neuron>());
		for(int j = 0; j <= topology[i]; j++){
			int fwd, bck;
			if(i == 0) fwd = topology[i+1], bck = 0;
			else if(i == topology.size()-1) fwd = 0, bck = topology[i-1];
			else fwd = topology[i+1], bck = topology[i-1];
			net.back().push_back(Neuron(fwd, bck, j));
		}
	}
	for(int i = 0; i < topology.size(); i++){
		net[i].back().OutputVal = 0;
	}

	f  = activation;
	df = derivative;
	AI::alpha = alpha;
	AI::eta = eta;
}

void AI::NeuralNetwork::FeedInData(const std::vector<double>& data){
	assert(data.size() == net[0].size()-1); //error checking
	
	for(int i = 0; i < data.size(); i++){
		net[0][i].OutputVal = data[i];
	}

	//do math!
	for(int i = 1; i < net.size(); i++){
		for(int j = 0; j < net[i].size(); j++){
			net[i][j].activate(net[i-1]);
		}
	}

}


void AI::NeuralNetwork::Backpropagation(const std::vector<double>& data){
	
	assert(data.size() == net.back().size() - 1);

	//calculate errors and gradients

	for(int i = 0; i < net.back().size() - 1; i++){
		net.back()[i].calculateOutputGradient(data[i]);
	}


	for(int i = net.size() - 2; i > 0; i--){
		for(int j = 0; j < net[i].size() - 1; j++){
			net[i][j].calculateHiddenLayerGradiend(net[i+1]);
		}
	}
	//adjusting weights
	for(int i = net.size() - 1; i > 0; i--)	{
		for(int j = 0; j < net[i].size() - 1; j++){
			net[i][j].updateWeights(net[i-1]);
		}
	}
}

void AI::NeuralNetwork::getData(std::vector<double>& data) const {
	data.clear();
	for(int i = 0; i < net.back().size(); i++){
		data.push_back(net.back()[i].OutputVal);
	}
}

void AI::NeuralNetwork::exportData(const std::string& outFile) const{

	std::fstream out(outFile, std::ios::out);
	out << net.size() << '\n';	//layers of neurons
	std::vector<connections> neuronWeights;

	for(int i = 0; i < net.size(); i ++){
		out << net[i].size() << ' '; //n/o neurons in that layer

		for(int j = 0; j < net[i].size(); j++){

			neuronWeights = net[i][j].getConnections();
			out << neuronWeights.size() << ' '; // n/o neurons of the next layer
												// AKA weights

			for(int k = 0; k < neuronWeights.size(); k++){
				out << neuronWeights[k].weight << ' '; //the weight itself
			}
		}
	}
	out << 0;
	out << alpha << ' ' << eta;
}

AI::NeuralNetwork::NeuralNetwork(const std::string& file,
								 std::function<long double(long double)> activation,
				  				 std::function<long double(long double)> derivative){
	std::fstream inp(file, std::ios::in);
	int n; inp >> n; //layers of neurons

	for(int i = 0; i < n; i++){
		net.push_back(std::vector<Neuron>());
		int neurons; inp >> neurons; // n/o neurons in the layer

		for(int j = 0; j < neurons; j++){
			int nweight; inp >> nweight; // n/o neurons in the next layer
			std::vector<double> w;

			for(int k = 0; k < nweight; k++){
				double weight; inp >> weight; w.push_back(weight);
			}
			
			net[i].push_back(Neuron(nweight, 0, j, 0, &w));

		}
	}
	f  = activation;
	df = derivative;
	
	inp >> AI::alpha >> AI::eta;


}

/************NEURON****************/


AI::Neuron::Neuron(int fwdElements, int bckElements, int i, bool randW,
				const std::vector<double>* v){
	ith = i;
	if(randW){
		std::random_device rd;
		std::mt19937 generator(rd());

		long double xavierFactor = 1.0/sqrt(fwdElements);
		
		for(int i = 0; i < fwdElements; i++){
			std::normal_distribution<double> dist(0, xavierFactor);
			long double w = dist(generator);
			fwd.push_back({w, 0});
		}
	}
	else
		for(int i = 0; i < fwdElements; i++){
			fwd.push_back({(*v)[i], 0});
		}
}

void AI::Neuron::updateWeights(std::vector<Neuron>& lastLayer){
	for(int i = 0; i < lastLayer.size() - 1; i++){
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
	for(int i = 0; i < lastLayer.size() - 1; i++){
		totalX += lastLayer[i].OutputVal*lastLayer[i].fwd[ith].weight;
	}
	totalX += bais;
	OutputVal = f(totalX);
// 	if(OutputVal == 0)
// 		std::cerr << totalX << '\n';
}

void AI::Neuron::calculateOutputGradient(double data){
	Gradient = (data - OutputVal) * df(OutputVal);
}

void AI::Neuron::calculateHiddenLayerGradiend(const std::vector<AI::Neuron>& nextLayer){
	long double sum = 0;
	for(int i = 0; i < nextLayer.size() - 1; i++){
		sum += fwd[i].weight * nextLayer[i].Gradient;
	}

	Gradient = sum * df(OutputVal);
}
