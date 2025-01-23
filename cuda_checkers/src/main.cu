// #include <iostream>
// #include <thread>
// #include <chrono>
// #include <cuda_runtime.h>
// #include <curand_kernel.h>

#include "../include/Game.h"

int main() {
    // Game game = Game::getGameInfo();
    Game game = Game();
    game.run();
    return 0;
}

// #include <iostream> 
// #include <cuda_runtime.h>

// int main() { 
//     int deviceCount = 0; 
//     cudaError_t err = cudaGetDeviceCount(&deviceCount); 
//     if (err != cudaSuccess) { 
//         std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl; 
//         return 1; 
//     } 
//     std::cout << "Number of CUDA-capable devices: " << deviceCount << std::endl; 
//     return 0;
// }