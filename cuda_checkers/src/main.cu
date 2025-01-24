#include <iostream>
#include <fstream>
#include <string>
#include "../include/Game.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <output_file>" << std::endl;
        return 1;
    }
    std::string outputFileName = argv[1];
    
    Game game = Game::getGameInfo(outputFileName);
    game.run();
    return 0;
}