
#ifndef BOARD_H
#define BOARD_H

#include <cstdint>
#include "Move.h"
#include <curand_kernel.h>
#include <random>

#define INITIAL_WHITE   0x00000FFF
#define INITIAL_BLACK   0xFFF00000
#define INITIAL_QUEENS  0x00000000

#define BOARD_SIZE      32

#define ODD_ROWS        0xF0F0F0F0
#define EVEN_ROWS       0x0F0F0F0F

#define ODD_UP_RIGHT    5
#define ODD_UP_LEFT     4
#define ODD_DOWN_RIGHT  3
#define ODD_DOWN_LEFT   4

#define EVEN_UP_RIGHT   4
#define EVEN_UP_LEFT    3
#define EVEN_DOWN_RIGHT 4
#define EVEN_DOWN_LEFT  5

#define UP_RIGHT_CAPT   9
#define UP_LEFT_CAPT    7
#define DOWN_RIGHT_CAPT 7
#define DOWN_LEFT_CAPT  9

#define FIRST_ROW       0x0000000F
#define LAST_ROW        0xF0000000
#define FIRST_COLUMN    0x01010101
#define SECOND_COLUMN   0x10101010
#define S_LAST_COLUMN   0x08080808
#define LAST_COLUMN     0x80808080

#define TIME_LIMIT_MS   1000
#define NUMBER_OF_GAMES 1000

#define MAX_MOVES 48

class Board {
public:
    uint32_t white;
    uint32_t black;
    uint32_t queens;
    bool whiteToMove;

    __host__ __device__ Board(bool whiteToMove = true) : white(INITIAL_WHITE), black(INITIAL_BLACK), 
                                                        queens(INITIAL_QUEENS), whiteToMove(whiteToMove) {}

    __host__ __device__ int generate_moves(Move *moves, Move *stackmem);
    __host__ int simulate_game_cpu(std::mt19937& rng, Move *moves, Move *stack);
    __host__ int simulate_n_games_cpu(std::mt19937& rng, Move *moves, Move *stack, int n = NUMBER_OF_GAMES, float time_limit_ms = TIME_LIMIT_MS);
    __device__ int simulate_game_gpu(curandState* state, Move *moves, Move *stack);
    __device__ int simulate_n_games_gpu(curandState* state, Move *moves, Move *stack, int n = NUMBER_OF_GAMES, float time_limit_ms = TIME_LIMIT_MS);
    __host__ __device__ Board apply_move(const Move &move);
    __host__ void print_board();
    __host__ void print_square(int row, int col);
    __host__ __device__ bool is_equal(Board board);

};

#endif // BOARD_H