
#ifndef BOARD_H
#define BOARD_H

#include <cstdint>
#include "Move.h"

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

struct Board {
public:
    uint32_t white;
    uint32_t black;
    uint32_t queens;
    bool whiteToMove;

    __host__ __device__ Board(bool whiteToMove = true) : white(INITIAL_WHITE), black(INITIAL_BLACK), 
                                                        queens(INITIAL_QUEENS), whiteToMove(whiteToMove) {}

    __host__ __device__ int generate_moves(Move *moves);
    __host__ __device__ int simulate_game();
    __host__ __device__ int simulate_n_games(int n);
    __host__ __device__ Board apply_move(const Move &move);
    __host__ void print_board();
    __host__ void print_square(int row, int col);

};

#endif // BOARD_H