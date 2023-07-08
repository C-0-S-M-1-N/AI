#include <cinttypes>
#include <iostream>
#include <fstream>
#include <vector>
#include <inttypes.h>
#include <algorithm>
#include <stack>
#include <NeuralNetwork/NeuralNetwork.hpp>

/*
 * This function had nothing to do with the AI, is used to read
 * the test cases attributes, you can find them on the site: http://yann.lecun.com/exdb/mnist/
 *
 * */
void read_data(std::fstream& img, std::fstream& lab,
				uint32_t& elements,
				uint32_t& rows, 
				uint32_t& columns){
	uint32_t magic;

	img.read(reinterpret_cast<char*>(&magic), sizeof(magic));
	magic = __builtin_bswap32(magic);
	if(magic != 2051) {exit(0);}

	lab.read(reinterpret_cast<char*>(&magic), sizeof(magic));
	magic = __builtin_bswap32(magic);
	if(magic != 2049) {exit(0);}


	img.read(reinterpret_cast<char*>(&elements), sizeof(elements)); elements 	= __builtin_bswap32(elements);
	lab.read(reinterpret_cast<char*>(&elements), sizeof(elements)); elements 	= __builtin_bswap32(elements);
	img.read(reinterpret_cast<char*>(&rows), sizeof(rows));			rows 	 	= __builtin_bswap32(rows);
	img.read(reinterpret_cast<char*>(&columns), sizeof(columns));	columns 	= __builtin_bswap32(columns);
	
}

size_t getPrediction(const std::vector<long double> prediction){
	size_t ret = 0;
	long double max = -10;
	for(size_t i = 0; i < prediction.size(); i++){
		if(prediction[i] > max){
			max = prediction[i];
			ret = i;
		}
	}
	return ret;
}

struct data_{
	std::vector<long double> img, correct;
	size_t digit;
};

int main(){
	srand(time(0)); // random seed
	
	// opening the training data sets
	std::fstream img("../data/train-images-idx3-ubyte", std::ios::in | std::ios::binary),
				 lab("../data/train-labels.idx1-ubyte", std::ios::in | std::ios::binary);
	
	// attributes about the data sets
	uint32_t elements;
	uint32_t rows, columns;
	unsigned char digit;
	unsigned char pixel;
	

	// reading the attributes
	read_data(img, lab, elements, rows, columns);

	std::vector<int> blueprint = {int(rows*columns), 50, 100, 50, 10}; // NN blueprint
	
	AI::NeuralNetwork nn(blueprint, AI::Functions::ReLU.activation, AI::Functions::ReLU.derivative, 0.9, 0.001);
	
	std::vector<data_> imgs(elements); // training data disposed in a vector so it can be 
									   // randomized
	for(int i = 0; i < elements; i++){ // loop over all the images

		for(int j = 0; j < rows*columns; j++){ // reading the image itself
			img.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
			imgs[i].img.push_back((double)pixel/255);
		}

		lab.read(reinterpret_cast<char*>(&digit), sizeof(digit)); // reads the digit that is represented in the image
		imgs[i].digit = digit;
		for(int j = 0; j < 10; j++) imgs[i].correct.push_back(j == digit ? 1.0 : 0.0); // make the correct output ahead of time
	}

	img.close();
	lab.close();

	std::cout << "training\n";
	std::random_shuffle(imgs.begin(), imgs.end());

// 	std::vector<std::vector<data_>> batch;
	std::stack<std::vector<data_>> batch;
	{
		std::vector<data_> aux;
		for(size_t i = 0; i < imgs.size(); i++){
			if(i && i % 2500){batch.push(aux); aux.clear();}
			aux.push_back(imgs[i]);
		}
		batch.push(aux);
	}
	
	const long double trash_hold = 88.5;
	const size_t batch_size = batch.size();
	std::vector<double long> result;
	while(!batch.empty()){
		size_t corrects = 0;
		while((long double)corrects/batch.top().size()*100 < trash_hold){
// 			std::cerr << (long double)corrects/batch.top().size()*100 << "%\n";
// 			std::cerr << corrects << '\n';	
			corrects = 0;
			
			for(auto i: batch.top()){
				nn.FeedInData(i.img);

				nn.getData(result);
				if(getPrediction(result) == i.digit){corrects++;

				}

				nn.Backpropagation(i.correct);	
			}
		}	
		std::cerr << batch_size-batch.size()+1 << "/" << batch_size << '\n'
				<< (long double)corrects/batch.top().size()*100 << "%\n";
		batch.pop();
	}

	
	nn.exportData("digitRecognition.nn"); // exports the NN status for using it in other programs
	
}
