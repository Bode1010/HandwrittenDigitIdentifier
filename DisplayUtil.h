#pragma once
#include <SFML/Graphics.hpp>
#include <iostream>

class Button
{
	int xLocation, yLocation, length, width;
public:
	Button(int xLocation, int yLocation, int lenght, int width);
	bool update(sf::RenderWindow& window);
};

