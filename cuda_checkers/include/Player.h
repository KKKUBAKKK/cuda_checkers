#ifndef PLAYER_H
#define PLAYER_H

#define THREADS_PER_BLOCK 256

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#include "Node.h"

__global__ void simulate_game_gpu_kernel(Board initial_board, float* results, curandState* states, bool is_player_white);
__global__ void init_curand(curandState* state, unsigned long seed);

class Player {
private:
    Node *root;
    float time_limit_ms;
    int max_games;
    int max_iterations;
    curandState* states;

public:
    bool is_white;
    bool is_cpu;

    explicit Player(bool is_white = true, bool is_cpu = true, 
    int max_games = MAX_GAMES, int max_iterations = MAX_ITERATIONS, 
    float time_limit_ms = TIME_LIMIT_MS);

    Player(Board board, bool is_white = true, bool is_cpu = true, 
    int max_games = MAX_GAMES, int max_iterations = MAX_ITERATIONS, 
    float time_limit_ms = TIME_LIMIT_MS);

    ~Player();

    int findEqualChild(Board board);
    void move_root(Board startBoard);
    Node* select();
    Node* expand(Node* node);
    float simulate(Node *node);
    float simulate_cpu(Board board);
    float simulate_gpu(Board board);
    void backpropagate(Node *node, float score);
    int mcts_loop();
    Node *choose_move();
    Board make_move(Board start_board);
};

#endif // PLAYER_H