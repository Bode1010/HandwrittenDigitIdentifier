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
//If I close program while running and save file is broken, open backup save file instead. REMEMBER TO DISABLE TRAINING BEFORE OPENING ANY SAVE FILE\
IF THE FILE IS CORRUPTED IT WILL OVERWRITE THE BACKUP IMMEDIATELY

//remove bias from convo and upsample layers
//smth is wrong with load and save


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

void ReadDataset(string fileName, vector<vector<float>>& dataset, int numItems) {
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

	numOfItems = numItems;
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

void ReadLabels(string fileName, vector<vector<float>>& labels, int numItems) {
	ifstream inFile(fileName, std::ios::in | std::ios::binary);
	int magicNum, numOfItems;
	inFile.read((char*)&magicNum, sizeof(int));
	magicNum = reverseInt(magicNum);
	inFile.read((char*)&numOfItems, sizeof(int));
	numOfItems = reverseInt(numOfItems);

	numOfItems = numItems;
	labels = vector<vector<float>>(numOfItems);

	unsigned char temp;
	for (int i = 0; i < numOfItems; i++) {
		inFile.read((char*)&temp, sizeof(char));
		labels[i] = OneHotEncodeOneToTen((int)temp);
	}

}

void autoEncoderTest() {
	//increasing num of filters/num of layers gives high training time/plateauing at high errors for some reason
	//Read dataset from file
	vector<vector<float>>* dataset = new vector<vector<float>>();
	ReadDataset("t10k-images.idx3-ubyte", *dataset, 100);

	Layer input = Util::Dense(28 * 28, NONE);
	Layer h1 = Util::Dense(50, RELU);
	Layer h2 = Util::Dense(15, RELU);
	Layer h3 = Util::Dense(30, RELU);
	Layer output = Util::Dense(28 * 28, SIGMOID);

	vector<Layer> Layout = { input, h1, output };
	NeuralNet myNet(Layout);
	myNet.setDebugFlag(true);
	myNet.setSaveFile("AutoEncoderNet3.hnn");
	bool load = true;
	if (load) {
		myNet.load("BackupAutoEncoderNet3.hnn");
	}
	else {
		//myNet.load("BackupAutoEncoderNet5.hnn");
		myNet.trainTillError(*dataset, *dataset, 10, 10000, 15);
	}

	myNet.feedForward((*dataset)[0]);
	myNet.BackPropagate((*dataset)[0], 0);
	cout << myNet.getError() << endl;

	/*******Draw to Screen**********/
	DisplayCnn artist(myNet, (*dataset), 28, 28);
	artist.Draw();
	/******************************/

	delete(dataset);
}

void handwrittenDigitIdentifierTest() {
	srand(0);
	//Read dataset from file. Images are 28 by 28
	vector<vector<float>>* dataset = new vector<vector<float>>();
	vector<vector<float>>* label = new vector<vector<float>>();
	ReadDataset("t10k-images.idx3-ubyte", *dataset, 100);
	ReadLabels("t10k-labels.idx1-ubyte", *label, 100);

	//Neural network architecture: Image is black and white so its initial depth is 1, afterwards its depth will be the num of filters from the prev layer
	Layer input = Util::Dense(28 * 28, NONE, 0, 1, 28 * 28);
	Layer h0 = Util::Upsample(28, 28, 1, 2, 2);
	Layer h1 = Util::Convo(3, 3, 8, RELU, false, 2, &h0);
	//Back to back conv layers don't need image size or previous layer depth specified if a pointer to previous convolutional layer is given
	Layer h2 = Util::Dense(75, RELU, 0, 1, 75);
	Layer output = Util::Dense(10, SOFTMAX, 0, 1, 10);

	vector<Layer> Layout = { input, h0, h1, h2, output };
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
}

int main() {
	//handwrittenDigitIdentifierTest();
	autoEncoderTest();

	std::cout << "Program Terminated" << std::endl;
	return 0;
}