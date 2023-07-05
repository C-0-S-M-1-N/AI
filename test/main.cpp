#include<iostream>
#include<fstream>
#include<vector>
#include <inttypes.h>
#include <algorithm>
#include <NeuralNetwork/NeuralNetwork.hpp>

typedef unsigned int uint_32;

void read_data(std::fstream& img, std::fstream& lab,
				uint_32& elements,
				uint_32& rows, 
				uint_32& columns){
	uint_32 magic;

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
// 	unsigned img[28*28], correct[10];
	std::vector<double> img, correct;
// 	unsigned correct;
};

uint64_t image_size = 28*28*60'000,	lab_size = 1;
uint64_t default_offset_img = 12, 	default_offset_lab = 7;

int main(){
	srand(time(0));
	std::fstream img("../data/train-images-idx3-ubyte", std::ios::in | std::ios::binary),
				 lab("../data/train-labels.idx1-ubyte", std::ios::in | std::ios::binary);
	uint_32 elements;
	uint_32 magic1, magic2;
	uint_32 rows, columns;
	unsigned char digit;
	unsigned char pixel;

	read_data(img, lab, elements, rows, columns);

	std::vector<int> blueprint = {int(rows*columns), 50, 100, 50, 10};

	AI::NeuralNetwork nn(blueprint);
	
	std::vector<data_> imgs(elements);
	for(int i = 0; i < elements; i++){
		for(int j = 0; j < rows*columns; j++){
			img.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
			imgs[i].img.push_back((double)pixel/255);
		}
		lab.read(reinterpret_cast<char*>(&digit), sizeof(digit));
		for(int j = 0; j < 10; j++) imgs[i].correct.push_back(j == digit ? 1.0 : 0.0);
// 		std::cerr << i << '\n';
	}
	img.close();
	lab.close();

	std::cout << "training\n";
// 	std::random_shuffle(imgs.begin(), imgs.end());
// 	
// 	for(int i = 0; i < 28; i++){
// 		for(int j = 0; j < 28; j++){
// 			std::cerr << (imgs[4].img[i*28+j] > 0.5 ? 1 : 0) << ' ';
// 		}
// 		std::cerr << '\n';
// 	}
// 	for(auto i : imgs[4].correct) std::cerr << i << ' ';
// return 0;

start_:
	int corrects = 0;
	std::random_shuffle(imgs.begin(), imgs.end());
	for(int i = 0; i < elements; i++){
		std::vector<double> result(10);
		
		nn.FeedInData(imgs[i].img);
		
		nn.getData(result);

		nn.Backpropagation(imgs[i].correct);
		for(int j = 0; j < 10; j++) if(imgs[i].correct[j] == 1){ digit = j; break; }
		double max = result[0];
		int guess = 0;
		for(int j = 1; j < 10; j++){ if(result[j] > max) {max = result[j]; guess = j;} }
		if(digit == guess){
			corrects ++;
		}
		if(i%10000 == 0){
			for(auto i:result) std::cout << i << ' ';
			std::cout << "\t " << guess << " " << (int)digit << '\n';
		}
// 		std::cerr << i << '\n';
	}

	if((double)corrects/elements*100 < 87.5) { std::cout << (double)corrects/elements*100 << '\n'; goto start_;}

	nn.exportData("digitRecognition.nn");
	
// 	img.open("./data/t10k-images-idx3-ubyte", std::ios::binary | std::ios::in);
// 	lab.open("./data/t10k-labels-idx1-ubyte", std::ios::binary | std::ios::in);
// 	read_data(img, lab, elements, rows, columns);
	
}
