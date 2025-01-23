#include <iostream>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "../include/Game.h"

int main() {
    // Game game = Game::getGameInfo();
    Game game = Game();
    game.run();
    return 0;
}