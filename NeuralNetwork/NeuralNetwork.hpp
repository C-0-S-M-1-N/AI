#pragma once
#include <vector>
#include <string>
#include <functional>
#include <inttypes.h>

#define EULER 2.71828

namespace AI{

namespace Functions{

struct functions{
	std::function<long double(long double)> activation, derivative;
};

extern functions sigmoid, tanh, ReLU, linear, PReLU,
	   			 binaryStep;
extern long double PReLU_argument;

}; // namespace Functions


struct connections{ long double weight, deltaweight; };

class Neuron{
	std::vector<connections> fwd;
	int ith;
	

public:
	
// 	static long double alpha, eta;
	long double OutputVal;
	long double Gradient;
	Neuron(int fwdElements, int bckElements, int i, bool randW = 1,
					const std::vector<double>* v = 0);
	void activate(const std::vector<Neuron>& lastLayer);
	void calculateOutputGradient(double data);
	void calculateHiddenLayerGradiend(const std::vector<Neuron>& nextLayer);
	void updateWeights(std::vector<Neuron>& lastLayer);
	std::vector<connections> getConnections() const { return fwd; }
};

extern long double eta, alpha;

class NeuralNetwork{
	
	std::vector<std::vector<Neuron>> net;
public:
	/**
	 * @breif network constructor
	 * @param takes the topology of the nn, number of neurons/layer
	 *
	 * */
	NeuralNetwork(const std::vector<int>& topology,
				  std::function<long double(long double)> activation = AI::Functions::linear.activation,
				  std::function<long double(long double)> derivative = AI::Functions::linear.derivative,
				  const long double alpha = 0.8, const long double eta = 0.0001);
	/**
	 * @breif function that provides the input layer data
	 * @param data witch is type T
	 * */
	void FeedInData(const std::vector<double>& data);
	/**
	 * @breif function that trains the nn based on the correct output
	 * @param the correct output
	 * */
	void Backpropagation(const std::vector<double>& outputData);
	/**
	 * @breif function that returns the data made by the nn 
	 * @param provider
	 * */
	void getData(std::vector<double>& data) const;

	void exportData(const std::string& outFile) const;

	NeuralNetwork(const std::string& file, 
				  std::function<long double(long double)> activation = AI::Functions::linear.activation,
				  std::function<long double(long double)> derivative = AI::Functions::linear.derivative
				 );
};




};
