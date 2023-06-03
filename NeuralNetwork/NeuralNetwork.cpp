#include "NeuralNetwork.hpp"
#include <cstdlib>
#include <random>
#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>


/***************NEURAL NETWORK***************/
AI::NeuralNetwork::NeuralNetwork(const std::vector<int>& topology){
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
// 	error = 0; //reset the error
// 	for(int i = 0; i < net.back().size() - 1; i++){
// 		error += (data[i] - net.back()[i].OutputVal)*(data[i] - net.back()[i].OutputVal);
// 	}
// 	error /= ((double)data.size()-1);
// 	error = std::sqrt(error);

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
}

AI::NeuralNetwork::NeuralNetwork(const std::string& file){
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

}

/************NEURON****************/

long double AI::Neuron::alpha = 0.8;
long double AI::Neuron::eta = 0.0001;

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
				eta * lastLayer[i].OutputVal * Gradient
				+alpha * lastLayer[i].fwd[ith].deltaweight
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

	Gradient = sum * AI::Neuron::df(OutputVal);
}

long double AI::Neuron::f(long double x){
	return std::max((double long)0.0, x);
}
long double AI::Neuron::df(long double x){
	return x >= 0 ? 1.0 : 0.0;
}


