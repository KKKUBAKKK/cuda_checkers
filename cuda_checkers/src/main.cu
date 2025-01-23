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