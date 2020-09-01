#include "NeuralNet.h"
#include "Display.h"
#include <fstream>
#include <chrono>

//Notes:
//The trainedIdentifier.hcnn save file trained on 30k images for 7 hours to achieve this level of mastery. DO NOT DELETE!
//Use square filters if not rotating the matrix will break. I could have accounted for this, but its easier (and more space efficient) to just use square filters
//During the conv feed forward at some point in the code it has one or two duplicates of an entire layer, this might be inefficient. If im ever pressured for space I can fix this
//If filters are becoming nan(ind), one of the matrixes updating them is all zeros, you might want to check convbackprop, nextLayerDense or nextLayerConv
//If the vizualization is too slow, its cuz im repeating calculations that only need to be done once in the vizualization process. Remember to take those out during refactoring

int reverseInt(int i) {
	char c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;
	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + (c4);
}

vector<float> OneHotEncodeOneToTen(int x) {
	vector<float> result(10);
	result[x] = 1;
	return result;
}

void ReadDataset(string fileName, vector<vector<float>>& dataset) {
	ifstream inFile(fileName, std::ios::in | std::ios::binary);
	int magicNum, numOfItems, rows, columns;
	inFile.read((char*)&magicNum, sizeof(int));
	magicNum = reverseInt(magicNum);
	inFile.read((char*)&numOfItems, sizeof(int));
	numOfItems = reverseInt(numOfItems);
	inFile.read((char*)&rows, sizeof(int));
	rows = reverseInt(rows);
	inFile.read((char*)&columns, sizeof(int));
	columns = reverseInt(columns);

	numOfItems = 100;
	dataset = vector<vector<float>>(numOfItems);

	unsigned char temp;
	for (int i = 0; i < numOfItems; i++) {
		for (int j = 0; j < columns; j++) {
			for (int k = 0; k < rows; k++) {
				inFile.read((char*)&temp, sizeof(char));
				dataset[i].push_back(((float)temp)/255.0f);
			}
		}
	}
}

void ReadLabels(string fileName, vector<vector<float>>& labels) {
	ifstream inFile(fileName, std::ios::in | std::ios::binary);
	int magicNum, numOfItems;
	inFile.read((char*)&magicNum, sizeof(int));
	magicNum = reverseInt(magicNum);
	inFile.read((char*)&numOfItems, sizeof(int));
	numOfItems = reverseInt(numOfItems);

	numOfItems = 100;
	labels = vector<vector<float>>(numOfItems);

	unsigned char temp;
	for (int i = 0; i < numOfItems; i++) {
		inFile.read((char*)&temp, sizeof(char));
		labels[i] = OneHotEncodeOneToTen((int)temp);
	}

}

int main() {
	srand(0);
	//Read dataset from file. Images are 28 by 28
	vector<vector<float>>* dataset = new vector<vector<float>>();
	vector<vector<float>>* label = new vector<vector<float>>();
	ReadDataset("t10k-images.idx3-ubyte", *dataset);
	ReadLabels("t10k-labels.idx1-ubyte", *label);

	//Neural network architecture: Image is black and white so its initial depth is 1, afterwards its depth will be the num of filters from the prev layer
	Layer input = Util::Dense(28 * 28, NONE, 0, 1, 28 * 28);
	Layer h1 = Util::Convo(3, 3, 8, RELU, 28, 28, 1, false);
	//Back to back conv layers don't need image size or previous layer depth specified if a pointer to previous convolutional layer is given
	Layer h2 = Util::Convo(5, 5, 12, RELU, false, 2, &h1);
	Layer h3 = Util::Dense(75, RELU, 0, 1, 75);
	Layer output = Util::Dense(10, SOFTMAX, 0, 1, 10);

	vector<Layer> Layout = { input, h1, h3, output };
	NeuralNet myNet(Layout);
	myNet.setDebugFlag(true); 

	//Train the network
	chrono::system_clock::time_point startTime = chrono::system_clock::now();
	myNet.trainTillError(*dataset, *label, 10, 10, .1);
	chrono::system_clock::time_point endTime = chrono::system_clock::now();
	std::chrono::duration<double, std::milli> timeTaken = (endTime - startTime)/1000.f;
	cout << "Time taken: " << timeTaken.count() << endl;

	//myNet.save("Identify7.hcnn");
	//myNet.load("TrainedIdentifier.hcnn");

	/*******Draw to Screen**********/
	DisplayCnn artist(myNet, (*dataset), 28, 28);
	artist.Draw();
	/******************************/
	
	delete(dataset);
	delete(label);

	std::cout << "Program Terminated" << std::endl;
	return 0;
}