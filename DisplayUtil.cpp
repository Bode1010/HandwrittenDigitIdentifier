#include "DisplayUtil.h"

Button::Button(int xLoc, int yLoc, int len, int wid) {
	xLocation = xLoc;
	yLocation = yLoc;
	length = len;
	width = wid;
}

bool Button::update(sf::RenderWindow& window) {
	auto mouseDim = sf::Mouse::getPosition(window);

	sf::Event evnt;
	if (window.pollEvent(evnt)) {
		switch (evnt.type) {
		case sf::Event::MouseButtonPressed:
			if (evnt.mouseButton.button == sf::Mouse::Left) {
				if (mouseDim.x > xLocation&& mouseDim.x < xLocation + length && mouseDim.y > yLocation&& mouseDim.y < yLocation + width) {
					std::cout << "Button clicked!" << std::endl;
					return true;
				}
			}
			break;
		}
	}
	return false;
}
