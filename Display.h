#pragma once
#include "NeuralNet.h"
#include "DisplayUtil.h"
#include <string>

//Display convolutional neural network. Only works with black and white images
class DisplayCnn
{
public:
	DisplayCnn() {};
	DisplayCnn(NeuralNet& myNet, int imgLen, int imgWid);
	DisplayCnn(NeuralNet& myNet, vector<vector<float>> inputData, int imageLen, int imageWid);
	//Used to visually represent the results of neural network training. Was not designed to be used during training due to performance issues
	void Draw();
private:
	//Holds the neural network we are drawings
	NeuralNet myNet;
	//Length and width of the input images
	int imgLen, imgWid;
	//Holds the total number of filters in all the convolutional layers in the network
	int sumOfFilters;
	//Holds the inputs we can display to screen. given during initialization
	vector<vector<float>> inputs;
	//The index of the last image we drew to screen
	int currImageIndex = 0;
	//Holds the maximum input index, or inputs.size();
	int maxInput = 0;
	//To find which pixels are drawn on when drawing on the screen
	float brushThickness = .8f;
	bool mouseHeld = false;
	//Holds the current image being draws to screen and displayed in the network
	vector<float> curImage;
	//Holds the font the text on screen is, initialized in constructor
	sf::Font font;
	//Holds the font size
	int fontSize = 15;

	//Main Functions:
	//Takes in a probability array and the limits within which its meant to be drawn and returns a vertex array containing the portrait representation of those probabilities
	vector<sf::Vertex> probabilityVisualizer(int xLoc, int yLoc, vector<float> probabilities, int xLimit, int yLimit);
	//Takes filters from neural network and draws them within xLimit and yLimit
	vector<sf::Vertex> FilterVisualizer(int xLoc, int yLoc, int xLimit, int yLimit);
	//Takes images from the input database, the limits within which its supposed to be drawn and a reference to a vector<float> that can be passed to the neural network
	vector<sf::Vertex> DrawableWindow( int xLoc, int yLoc, bool leftClick, bool rightClick, bool randClick, bool clearClick, vector<float>& currentImage, int xLimit, int yLimit);
	//Takes in the number of probabilities as well as the info about the space its meant to be drawn in and spits out a vector of text numbering the probability
	vector<sf::Text> probabilityIndexVisualizer(int xLoc, int yLoc, int numOfProbabilities, int xLimit, int yLimit);
	//
	vector<sf::Vertex> outputImageVisualizer(int xLoc, int yLoc, int imgXDimension, int imgYDimension, int imgXLimit, int imgYLimit);

	//Auxiliary Functions:
	//Takes a vector<float> image, its dimensions and the limits in which its meant to be drawn and returns a vertex array that draws it
	vector<sf::Vertex> myImageToOnScreenRepresentation(vector<float>, int xDim, int yDim, int xLimit, int yLimit);
	//Takes in the number of filters and the limits of the space and returns how many lines need to be drawn on the x and y axis to split it evenly
	vector<int> splitSpace(int numOfItems, int limitX, int limitY);
};

