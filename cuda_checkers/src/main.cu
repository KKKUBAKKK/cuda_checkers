#include <iostream>
#include "../include/Board.h" // Include the header file where Board is defined
#include <thread>
#include <chrono>

#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void simulate_kernel(Board board, int* result, float time_limit_ms, int num_games) {
    // One thread only
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Initialize random state
        curandState state;
        curand_init(clock64(), 0, 0, &state);
        
        // Allocate moves and stack in thread-local memory
        Move moves[MAX_MOVES];
        Move stack[MAX_MOVES];
        
        // Run simulation
        *result = board.simulate_n_games_gpu(&state, moves, stack, num_games, time_limit_ms);
    }
}

int main() {
    // Trying out simulate_n_games_gpu
    // Setup board
    Board board;
    board.print_board();

    // Allocate device memory
    Board* d_board;
    int *d_result;
    cudaMalloc(&d_board, sizeof(Board));
    cudaMalloc(&d_result, sizeof(int));
    
    // Launch kernel with 1 thread
    simulate_kernel<<<1,1>>>(board, d_result, 1000.0f, 1000);
    
    // Copy board to device
    cudaMemcpy(d_board, &board, sizeof(Board), cudaMemcpyHostToDevice);

    // Get result
    int host_result;
    cudaMemcpy(&host_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "GPU Simulation score: " << host_result << std::endl;
    
    // Cleanup
    cudaFree(d_board);
    cudaFree(d_result);
    
    // return 0;

    // Trying out simulate_n_games_cpu
    // Initialize board and random generator
    // Board board;
    board = Board();
    std::random_device rd;
    std::mt19937 rng(rd());

    // Allocate move arrays
    Move moves[144];  // Max possible moves
    Move stack[144];  // Stack memory for move generation

    // Run simulations
    int num_games = 1000;
    float time_limit_ms = 1000.0f;
    int score = board.simulate_n_games_cpu(rng, moves, stack, num_games, time_limit_ms);

    // Print results
    std::cout << "CPU Simulation score: " << score << std::endl;
    std::cout << "Positive score favors white, negative favors black" << std::endl;

    return 0;
}