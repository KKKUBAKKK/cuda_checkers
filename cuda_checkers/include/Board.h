
#ifndef BOARD_H
#define BOARD_H

#include <cstdint>
#include "../../../../../usr/local/cuda-12.6/targets/x86_64-linux/include/crt/host_defines.h"

#define INITIAL_WHITE   0x00000FFF
#define INITIAL_BLACK   0xFFF00000
#define INITIAL_QUEENS  0x00000000

#define BOARD_SIZE      32

#define ODD_ROWS        0xF0F0F0F0
#define EVEN_ROWS       0x0F0F0F0F

#define ODD_UP_RIGHT    5
#define ODD_UP_LEFT     4
#define ODD_DOWN_RIGHT  (-3)
#define ODD_DOWN_LEFT   (-4)

#define EVEN_UP_RIGHT   4
#define EVEN_UP_LEFT    3
#define EVEN_DOWN_RIGHT (-4)
#define EVEN_DOWN_LEFT  (-5)

#define UP_RIGHT_CAPT   9
#define UP_LEFT_CAPT    7
#define DOWN_RIGHT_CAPT (-7)
#define DOWN_LEFT_CAPT  (-9)

#define FIRST_ROW       0x0000000F
#define LAST_ROW        0xF0000000
#define FIRST_COLUMN    0x11111111
#define SECOND_COLUMN   0x22222222
#define S_LAST_COLUMN   0x44444444
#define LAST_COLUMN     0x88888888

struct Board {
public:
    uint32_t white;
    uint32_t black;
    uint32_t queens;
    bool whiteToMove;

    __host__ __device__ Board(bool whiteToMove = true) : white(INITIAL_WHITE), black(INITIAL_BLACK), 
                                                        queens(INITIAL_QUEENS), whiteToMove(whiteToMove) {}

    __host__ __device__ void generate_moves();
    // __host__ __device__ bool is_even_row(int position);
};

#endif // BOARD_H