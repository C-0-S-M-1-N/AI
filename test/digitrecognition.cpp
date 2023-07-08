#include <iostream>
#include <fstream>
#include <numeric>
#include <inttypes.h>
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

int main(){
	std::fstream img("../data/t10k-images-idx3-ubyte", std::ios::in | std::ios::binary), // test files
				 lab("../data/t10k-labels-idx1-ubyte", std::ios::in | std::ios::binary);

	// attributes about the data sets
	uint32_t elements, rows, columns;	
	uint8_t digit, pixel;

	//reading the attributes
	read_data(img, lab, elements, rows, columns);
	
	//making the NN based on a save file, just give it the path to where it lies
	AI::NeuralNetwork nn("./digitRecognition.nn", AI::Functions::ReLU.activation, AI::Functions::ReLU.derivative);

	int correct = 0;
	std::cout << "testing . . .\n";
	for(int i = 0; i < elements; i++){
		//image reading
		std::vector<long double> image(rows*columns);
		for(int j = 0; j < rows*columns; j++){
			img.read(reinterpret_cast<char*>(&pixel), sizeof(pixel)); 
			image[j] = (long double)pixel/255;
		}
		lab.read(reinterpret_cast<char*>(&digit), sizeof(digit)); 
		std::vector<long double> result(10);

		nn.FeedInData(image);
		
		nn.getData(result);

// 		nn.Backpropagation(ll);  // no need to backpropagate
		

		double max = result[0];
		int guess = 0;

		for(int j = 1; j < 10; j++){ if(result[j] > max) {max = result[j]; guess = j;} } // get the AI guess

		if(guess == digit){ //track the accuracy of the AI
			correct++;
		}


	}

	std::cout << (double)correct/elements*100 << "%\n"; // output the accuracy

}
