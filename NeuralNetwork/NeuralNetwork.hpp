#pragma once
#include <vector>
#include <string>

#define EULER 2.71828

namespace AI{

struct connections{ double weight, deltaweight; };

class Neuron{
	std::vector<connections> fwd;
	static double f(double x);
	static double df(double x);
	int ith;
	
	static double alpha, eta;

public:
	double OutputVal;
	double Gradient;
	Neuron(int fwdElements, int bckElements, int i, bool randW = 1,
					const std::vector<double>* v = 0);
	void activate(const std::vector<Neuron>& lastLayer);
	void calculateOutputGradient(double data);
	void calculateHiddenLayerGradiend(const std::vector<Neuron>& nextLayer);
	void updateWeights(std::vector<Neuron>& lastLayer);
	std::vector<connections> getConnections() const { return fwd; }
};

class NeuralNetwork{
	
	std::vector<std::vector<Neuron>> net;
	double error; //RMS error type
	int counter;
public:
	/**
	 * @breif network constructor
	 * @param takes the topology of the nn, number of neurons/layer
	 *
	 * */
	NeuralNetwork(const std::vector<int>&);
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

	NeuralNetwork(const std::string& file);

};




};
