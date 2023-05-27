#include <iostream>
#include <fstream>
#include "./NeuralNetwork/NeuralNetwork.hpp"


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
	std::fstream img("./data/t10k-images-idx3-ubyte", std::ios::in | std::ios::binary),
				 lab("./data/t10k-labels-idx1-ubyte", std::ios::in | std::ios::binary);
	uint32_t elements, magic, rows, columns;	
	uint8_t digit, pixel;
	read_data(img, lab, elements, rows, columns);
	
	AI::NeuralNetwork nn("digitRecognition.nn");

	int correct = 0;
	for(int i = 0; i < elements; i++){
		//image reading
		std::vector<double> image(rows*columns);
		for(int j = 0; j < rows*columns; j++){
			img.read(reinterpret_cast<char*>(&pixel), sizeof(pixel)); 
			image[j] = (double)pixel/255;
		}
		lab.read(reinterpret_cast<char*>(&digit), sizeof(digit)); 
		std::vector<double> ll(10);
		std::vector<double> result(10);
		for(int j = 0; j < 10; j++){
			if(j == digit) ll[j] = 1.0;
			else ll[j] = 0.0;
		}

		nn.FeedInData(image);
		
		nn.getData(result);

		nn.Backpropagation(ll);
		

		double max = result[0];
		int guess = 0;
		for(int j = 1; j < 10; j++){ if(result[j] > max) {max = result[j]; guess = j;} }
		if(guess == digit){
			correct++;
		}


	}

	std::cout << (double)correct/elements*100 << "%\n";

}
