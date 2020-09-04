#include "Display.h"

DisplayCnn::DisplayCnn(NeuralNet& net, int imgLength, int imgWidth) {
	myNet = net;
	for (int i = 0; i < net.size(); i++) {
		sumOfFilters += net.getConvLayerFilterSize(i);
	}
	imgLen = imgLength;
	imgWid = imgWidth;
	curImage = vector<float>(imgLen * imgWid);

	if (!font.loadFromFile("Roboto-Black.ttf")) {
		cout << "Couldnt load font, check diplayCnn constructor" << endl;
	}
}

DisplayCnn::DisplayCnn(NeuralNet& net, vector<vector<float>> inputData, int imgLength, int imgWidth) {
	myNet = net;
	inputs = inputData;
	imgLen = imgLength;
	imgWid = imgWidth;
	for (int i = 0; i < net.size(); i++) {
		sumOfFilters+= net.getConvLayerFilterSize(i);
	}
	maxInput = inputData.size();
	curImage = vector<float>(imgLen * imgWid);

	if (!font.loadFromFile("Roboto-Black.ttf")) {
		cout << "Couldnt load font, check diplayCnn constructor" << endl;
	}
}

void DisplayCnn::Draw() {
	sf::RenderWindow window(sf::VideoMode(1024, 512), "Handwritten Digits Identifying Neural Network");
	//Visual locations and sizes
	int drawXLoc = 25;
	int drawYLoc = 100;
	int drawXLimit = 200;
	int drawYLimit = 200;

	//Auxiliary vars
	bool mouseHeld = false;

	//Text locations and sizes
	sf::Text buttonKey1;
	buttonKey1.setFont(font);
	buttonKey1.setString(" - Prev Image.");
	buttonKey1.setCharacterSize(10);
	buttonKey1.setPosition(sf::Vector2f(55, 400));
	sf::Text buttonKey2;
	buttonKey2.setFont(font);
	buttonKey2.setString(" - Rand Image.");
	buttonKey2.setCharacterSize(10);
	buttonKey2.setPosition(sf::Vector2f(55, 425));
	sf::Text buttonKey3;
	buttonKey3.setFont(font);
	buttonKey3.setString(" - Clear Image.");
	buttonKey3.setCharacterSize(10);
	buttonKey3.setPosition(sf::Vector2f(55, 450));
	sf::Text buttonKey4;
	buttonKey4.setFont(font);
	buttonKey4.setString(" - Next Image.");
	buttonKey4.setCharacterSize(10);
	buttonKey4.setPosition(sf::Vector2f(55, 475));


	//Button locations and dimensions
	int buttonWid = 15;
	int buttonLen = 25;
	int buttonX = 68;
	int buttonY = 300;

	sf::VertexArray buttons(sf::Quads, 32);
	//Button rectangles to be drawn to screen
	vector<sf::Color> colorArr = { sf::Color::Yellow, sf::Color::Green, sf::Color::Cyan, sf::Color::Magenta };
	for (int i = 0; i < 4; i++) {
		buttons[i*4].position = sf::Vector2f(25, 400 + (i * buttonLen));
		buttons[i*4 + 1].position = sf::Vector2f(25 + buttonLen, 400 + (i * buttonLen));
		buttons[i*4 + 2].position = sf::Vector2f(25 + buttonLen, 400 + (i * buttonLen) + buttonWid);
		buttons[i*4 + 3].position = sf::Vector2f(25, 400 + (i * buttonLen) + buttonWid);
		buttons[i * 4].color = colorArr[i];
		buttons[i*4 + 1].color = colorArr[i];
		buttons[i*4 + 2].color = colorArr[i];
		buttons[i*4 + 3].color = colorArr[i];
	}
	for (int i = 4; i < 8; i++) {
		buttons[i * 4 + 0].position = sf::Vector2f((i - 4) * (buttonLen + 5) + buttonX, buttonY);
		buttons[i * 4 + 1].position = sf::Vector2f((i - 4) * (buttonLen + 5) + buttonX + buttonLen, buttonY);
		buttons[i * 4 + 2].position = sf::Vector2f((i - 4) * (buttonLen + 5) + buttonX + buttonLen, buttonY + buttonWid);
		buttons[i * 4 + 3].position = sf::Vector2f((i - 4) * (buttonLen + 5) + buttonX, buttonY + buttonWid);
		buttons[i * 4].color = colorArr[i-4];
		buttons[i * 4 + 1].color = colorArr[i-4];
		buttons[i * 4 + 2].color = colorArr[i-4];
		buttons[i * 4 + 3].color = colorArr[i-4];
	}
	
	
	//myNet.getLayerSize(myNet.size()-1)
	auto probabilityIndexes = probabilityIndexVisualizer(855, 100, myNet.getLayerSize(myNet.size() - 1), 50, 350);
	//The visualizers given the info they need to vizualize(ie create images that can be displayed to screen)
	vector<sf::Vertex> t1, t2, t3, t4;
	myNet.feedForward(curImage);
	t1 = DrawableWindow(25, 100, 0, 0, 0, 0, curImage, 200, 200);
	t2 = FilterVisualizer(250, 50, 600, 400);
	t3 = probabilityVisualizer(900, 100, myNet.getOutput(), 50, 312);
	t4 = outputImageVisualizer(700, 100, imgLen, imgWid, 200, 200);

	while (window.isOpen()) {
		sf::Event evnt = sf::Event();
		
		while (window.pollEvent(evnt)) {
			switch (evnt.type) {
			case sf::Event::Closed:
				window.close();
				break;
			//Drawing on the window
			case sf::Event::MouseButtonPressed:
				if (evnt.mouseButton.button == sf::Mouse::Left) {
					mouseHeld = true;
					auto mouseDim = sf::Mouse::getPosition(window);
					if (mouseDim.x > drawXLoc&& mouseDim.x < drawXLoc + drawXLimit && mouseDim.y > drawYLoc&& mouseDim.y < drawYLoc + drawYLimit) {
						int xQuad = drawXLimit / imgLen;
						int yQuad = drawYLimit / imgWid;
						float basicDist = sqrt(xQuad * xQuad + yQuad * yQuad);
						//For every quad in the image, check if its within brush thickness, if it is, change its color value to 255
						for (int i = 0; i < curImage.size(); i++) {
							int xQuadLoc = (i % imgLen) * xQuad;
							int yQuadLoc = (i / imgLen) * yQuad;
							if (sqrt((mouseDim.x - xQuadLoc - drawXLoc) * (mouseDim.x - xQuadLoc - drawXLoc) + (mouseDim.y - yQuadLoc - drawYLoc) * (mouseDim.y - yQuadLoc - drawYLoc)) < brushThickness * basicDist) {
								curImage[i] = 1;
							}
						}

						t1 = myImageToOnScreenRepresentation(curImage, imgLen, imgWid, drawXLimit, drawYLimit);
						for (int i = 0; i < t1.size(); i++) {
							t1[i].position.x += drawXLoc;
							t1[i].position.y += drawYLoc;
						}
					}
				}
				break;
			//Clicking a button
			case sf::Event::MouseButtonReleased: 
				if (evnt.mouseButton.button == sf::Mouse::Left) {
					mouseHeld = false;
					auto mouseDim = sf::Mouse::getPosition(window);
					for (int i = 4; i < 8; i++) {
						if (mouseDim.x > buttons[i * 4].position.x && mouseDim.x < buttons[i * 4 + 1].position.x && mouseDim.y > buttons[i * 4].position.y&& mouseDim.y < buttons[i * 4 + 2].position.y) {
							switch (i) {
							case 4:
								t1 = DrawableWindow(25, 100, 1, 0, 0, 0, curImage, 200, 200);
								break;
							case 5:
								t1 = DrawableWindow(25, 100, 0, 0, 1, 0, curImage, 200, 200);
								break;
							case 6:
								t1 = DrawableWindow(25, 100, 0, 0, 0, 1, curImage, 200, 200);
								break;
							case 7:
								t1 = DrawableWindow(25, 100, 0, 1, 0, 0, curImage, 200, 200);
								break;
							}
						}
					}
					myNet.feedForward(curImage);
					t1 = DrawableWindow(25, 100, 0, 0, 0, 0, curImage, 200, 200);
					t2 = FilterVisualizer(250, 50, 600, 400);
					t3 = probabilityVisualizer(900, 100, myNet.getOutput(), 50, 312);
					t4 = outputImageVisualizer(700, 100, imgLen, imgWid, 200, 200);
				}
			default:
				//If we are holding the mouse down, draw to screen
				if (mouseHeld) {
					auto mouseDim = sf::Mouse::getPosition(window);
					if (mouseDim.x > drawXLoc&& mouseDim.x < drawXLoc + drawXLimit && mouseDim.y > drawYLoc&& mouseDim.y < drawYLoc + drawYLimit) {
						int xQuad = drawXLimit / imgLen;
						int yQuad = drawYLimit / imgWid;
						float basicDist = sqrt(xQuad * xQuad + yQuad * yQuad);
						//For every quad in the image, check if its within brush thickness, if it is, change its color value to 255
						for (int i = 0; i < curImage.size(); i++) {
							int xQuadLoc = (i % imgLen) * xQuad;
							int yQuadLoc = (i / imgLen) * yQuad;
							if (sqrt((mouseDim.x - xQuadLoc - drawXLoc) * (mouseDim.x - xQuadLoc - drawXLoc) + (mouseDim.y - yQuadLoc - drawYLoc) * (mouseDim.y - yQuadLoc - drawYLoc)) < brushThickness * basicDist) {
								curImage[i] = 1;
							}
						}

						t1 = myImageToOnScreenRepresentation(curImage, imgLen, imgWid, drawXLimit, drawYLimit);
						for (int i = 0; i < t1.size(); i++) {
							t1[i].position.x += drawXLoc;
							t1[i].position.y += drawYLoc;
						}
					}
				}
			}
		}
		window.clear(sf::Color(25, 100, 25));
		window.draw(buttons);
		window.draw(&t1[0], t1.size(), sf::Quads);
		//window.draw(&t2[0], t2.size(), sf::Quads);
		//window.draw(&t3[0], t3.size(), sf::Quads);
		window.draw(&t4[0], t4.size(), sf::Quads);
		for (int i = 0; i < probabilityIndexes.size(); i++) {
			//window.draw(probabilityIndexes[i]);
		}
		window.draw(buttonKey1);
		window.draw(buttonKey2);
		window.draw(buttonKey3);
		window.draw(buttonKey4);
		window.display();
	}
}

vector<sf::Vertex> DisplayCnn::myImageToOnScreenRepresentation(vector<float> image, int xDim, int yDim, int xLimit, int yLimit) {
	//If the image is longer than it is tall, adjust the yLimit to the nearest whole num, else, adjust the x
	if (yDim > xDim) {
		float yLim = 1.f * xLimit / xDim * yDim;
		yLimit = (yLim > (int)yLim) ? yLim + 1 : yLim;
	}
	else {
		float xLim = 1.f * yLimit / yDim * xDim;
		xLimit = (xLim > (int)xLim) ? xLim + 1 : xLim;
	}

	//Now that the limits and the image dimension are at the same ration, calculate the size of each pixel
	int pixelX = 1.f * xLimit / xDim;
	int pixelY = 1.f * yLimit / yDim;

	//Using the size of each pixel, transfer the image to a drawable format
	vector<sf::Vertex> result;
	for (int i = 0; i < yDim; i++) {
		for (int j = 0; j < xDim; j++) {
			sf::Color thisCol = sf::Color(image[i * xDim + j] * 255, image[i * xDim + j] * 255, image[i * xDim + j] * 255);
			result.push_back(sf::Vertex(sf::Vector2f(j * pixelX, i * pixelY), thisCol));
			result.push_back(sf::Vertex(sf::Vector2f(j * pixelX + pixelX, i * pixelY), thisCol));
			result.push_back(sf::Vertex(sf::Vector2f(j * pixelX + pixelX, i * pixelY + pixelY), thisCol));
			result.push_back(sf::Vertex(sf::Vector2f(j * pixelX, i * pixelY + pixelY), thisCol));
		}
	}

	return result;
}

vector<int> DisplayCnn::splitSpace(int num, int limx, int limy) {
	float ratio = 1.f * limx / limy;
	float minDist = abs(ratio - 1.f/num);
	vector<int> result = { 1, num };
	for (int i = 1; i <= num; i++) {//x
		for (int j = 1; j <= num; j++) {//y
			if (i * j == num) {
				float temp = 1.f * i / j;
				if (abs(ratio - temp) <= minDist) {
					result = { i-1 , j-1 };
				}
				else {
					return result;
				}
				break;
			}

		}
	}
	return result;
}

vector<sf::Vertex> DisplayCnn::probabilityVisualizer(int xLoc, int yLoc, vector<float> probabilities, int xLimit, int yLimit) {
	int xPadding = 2;
	int yPadding = 4;
	xLimit -= 2 * xPadding;
	yLimit -= 2 * yPadding;
	int width = (yLimit) / probabilities.size();
	
	vector<sf::Vertex> result;
	sf::Color thisCol = sf::Color(0, 255, 100);
	for (int i = 0; i < probabilities.size(); i++) {
		result.push_back(sf::Vertex(sf::Vector2f(xPadding + xLoc, i*(yPadding + width) + yLoc), thisCol));
		result.push_back(sf::Vertex(sf::Vector2f(xPadding + xLoc + probabilities[i] * xLimit, i*(yPadding + width) + yLoc), thisCol));
		result.push_back(sf::Vertex(sf::Vector2f(xPadding + xLoc + probabilities[i] * xLimit, i*(yPadding + width) + yLoc + width), thisCol));
		result.push_back(sf::Vertex(sf::Vector2f(xPadding + xLoc, i*(yPadding + width) + yLoc + width), thisCol));
	}
	return result;
}

vector<sf::Vertex> DisplayCnn::FilterVisualizer(int xLoc, int yLoc, int xLimit, int yLimit) {
	//Split space evenly (only needs to be done once)
	auto temp = splitSpace(sumOfFilters, xLimit, yLimit);
	int xLines = temp[0];
	int yLines = temp[1];

	//Move limits around so I get only square images of x and y dimensions with padding(only needs to be done once)
	if (yLines > xLines) {
		xLimit = 1.f * yLimit / (yLines+1) * (xLines+1);
	}
	else {
		yLimit = 1.f * xLimit / (xLines + 1) * (yLines + 1);
	}

	//turn all layer outputs into seperate images of the same (x, y) dimensions
	//Vector containing all the processed images
	vector<vector<float>> outputs;
	//Vector containing all the processed image dimensions
	vector<vector<int>> outputDims;
	for (int i = 0; i < myNet.size(); i++) {
		vector<float> imgPack = myNet.getLayerOutput(i);
		if (myNet.getConvLayerImgSize(i)[0] > 0) {
			for (int j = 0; j < myNet.getConvLayerFilterSize(i); j++) {
				outputDims.push_back({ myNet.getConvLayerImgSize(i)[0] , myNet.getConvLayerImgSize(i)[1] });
				outputs.push_back(vector<float>());
				for (int k = 0; k < myNet.getConvLayerImgSize(i)[0] * myNet.getConvLayerImgSize(i)[1]; k++) {
					outputs.back().push_back(imgPack[j * (myNet.getConvLayerImgSize(i)[0] * myNet.getConvLayerImgSize(i)[1]) + k]);
				}
			}
		}
	}

	//Combine all seperate images into one image
	vector<vector<sf::Vertex>> tempResult(sumOfFilters);
	for (int i = 0; i < sumOfFilters; i++) {
		tempResult[i] = myImageToOnScreenRepresentation(outputs[i], outputDims[i][0], outputDims[i][1], xLimit / (xLines + 1), yLimit/(yLines + 1));
	}

	vector<sf::Vertex> result;
	for (int i = 0; i < sumOfFilters; i++) {
		int offsetX = i%(xLines+1) * (xLimit/(xLines+1));
		int offsety = i/(xLines+1)*(yLimit/(yLines+1));
		for (int j = 0; j < tempResult[i].size(); j++) {
			sf::Vertex v = tempResult[i][j];
			v.position.x += offsetX + xLoc;
			v.position.y += offsety + yLoc;
			result.push_back(v);
		}
	}
	return result;
}

vector<sf::Vertex> DisplayCnn::DrawableWindow(int xLoc, int yLoc, bool leftClick, bool rightClick, bool randClick, bool clearClick, vector<float>& currentImage, int xLimit, int yLimit) {
	if (inputs.size() > 0) {
		if (rightClick) {
			if (currImageIndex < maxInput - 1) {
				currImageIndex++;
			}
			curImage = inputs[currImageIndex];
			auto temp = myImageToOnScreenRepresentation(curImage, imgLen, imgWid, xLimit, yLimit);
			for (int i = 0; i < temp.size(); i++) {
				temp[i].position.x += xLoc;
				temp[i].position.y += yLoc;
			}
			return temp;

		}
		else if (leftClick) {
			if (currImageIndex > 0) {
				currImageIndex--;
			}
			curImage = inputs[currImageIndex];
			auto temp = myImageToOnScreenRepresentation(curImage, imgLen, imgWid, xLimit, yLimit);
			for (int i = 0; i < temp.size(); i++) {
				temp[i].position.x += xLoc;
				temp[i].position.y += yLoc;
			}
			return temp;
		}
		else if (randClick) {
			currImageIndex = rand() % maxInput;
			curImage = inputs[currImageIndex];
			auto temp = myImageToOnScreenRepresentation(curImage, imgLen, imgWid, xLimit, yLimit);
			for (int i = 0; i < temp.size(); i++) {
				temp[i].position.x += xLoc;
				temp[i].position.y += yLoc;
			}
			return temp;

		}
	}
	if (clearClick) {
		vector<float> temp;
		for (int i = 0; i < imgLen * imgWid; i++) {
			temp.push_back(0);
		}
		curImage = temp;
		auto tmp = myImageToOnScreenRepresentation(curImage, imgLen, imgWid, xLimit, yLimit);
		for (int i = 0; i < tmp.size(); i++) {
			tmp[i].position.x += xLoc;
			tmp[i].position.y += yLoc;
		}
		return tmp;
	}

	//If somehow it gets here, return the current image
	auto temp = myImageToOnScreenRepresentation(currentImage, imgLen, imgWid, xLimit, yLimit);
	for (int i = 0; i < temp.size(); i++) {
		temp[i].position.x += xLoc;
		temp[i].position.y += yLoc;
	}
	return temp;
}

vector<sf::Vertex> DisplayCnn::outputImageVisualizer(int xLoc, int yLoc, int imgXDimension, int imgYDimension, int imgXLimit, int imgYLimit) {
	auto temp = myImageToOnScreenRepresentation(myNet.getOutput(), imgXDimension, imgYDimension, imgXLimit, imgYLimit);
	for (int i = 0; i < temp.size(); i++) {
		temp[i].position.x += xLoc;
		temp[i].position.y += yLoc;
	}
	return temp;
}

vector<sf::Text> DisplayCnn::probabilityIndexVisualizer(int xLoc, int yLoc, int probSize, int xLimit, int yLimit) {
	int xDim = xLimit / probSize;
	int yDim = yLimit / probSize;
	//cout << yDim << endl;

	vector<sf::Text> result;
	for (int i = 0; i < probSize; i++) {
		sf::Text temp;
		temp.setFont(font);
		temp.setCharacterSize(fontSize);
		string tempString = to_string(i) + ": ";
		temp.setString(tempString);
		temp.setPosition(sf::Vector2f(xLoc, yDim * i + yLoc ));
		result.push_back(temp);
	}
	return result;
}
