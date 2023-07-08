#include<iostream>
#include<fstream>
#include<vector>
#include <inttypes.h>
#include <algorithm>
#include <NeuralNetwork/NeuralNetwork.hpp>

const double MIN_ACCURACY = 87.5;


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

struct data_{
	std::vector<long double> img, correct;
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
		for(int j = 0; j < 10; j++) imgs[i].correct.push_back(j == digit ? 1.0 : 0.0); // make the correct output ahead of time
	}

	img.close();
	lab.close();

	std::cout << "training\n";
	size_t accuracy = 0;

	while((long double)accuracy/elements*100 < MIN_ACCURACY){
	accuracy = 0;

	std::random_shuffle(imgs.begin(), imgs.end());

	for(int i = 0; i < elements; i++){
		std::vector<long double> result(10); // making the result vector
		
		{ 
			nn.FeedInData(imgs[i].img);
			
			nn.getData(result);

			nn.Backpropagation(imgs[i].correct);

			/*
			 * 	feed the data into the NN
			 *
			 * 	get the data out 
			 *
			 * 	backpropagate
			 * */
		}
		

		for(int j = 0; j < 10; j++) if(imgs[i].correct[j] == 1){ digit = j; break; } // takes the digit that is supposed to be

		double max = result[0];
		int guess = -1;

		for(int j = 0; j < 10; j++){ if(result[j] >= max) {max = result[j]; guess = j;} } // takes NN's guess as what digit it is
																						 //
		if(digit == guess && max != 0){ // keep an eye for how the NN is doing
			accuracy ++;
		}

		if(i%10000 == 0){ // some console log that displays where it is <optional>
			for(auto i:result) std::cout << i << ' ';
			std::cout << "\t " << guess << " " << (int)digit << '\n';
		}

	}
	}

	std::cout << (double)accuracy/elements*100 << '\n'; 

	nn.exportData("digitRecognition.nn"); // exports the NN status for using it in other programs
	
}
