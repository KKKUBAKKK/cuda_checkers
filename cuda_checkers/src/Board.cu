#include "../include/Board.h"
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <../include/Move.h>
#include <../include/Stack.h>

#define MAX_MOVES 144

// __host__ __device__ bool Board::is_even_row(int position) {
//     return (position / 4) % 2 == 0;
// }

__host__ __device__ void Board::generate_moves() {
	Move moves[MAX_MOVES];
    int num_moves = 0;

    uint32_t player = whiteToMove ? white : black;
    uint32_t opponent = whiteToMove ? black : white;
    uint32_t all_pieces = white | black;

    Stack stack;

    // First try to find all captures (forced captures)
    // Find all pieces that can capture
    for (int i = 0; i < BOARD_SIZE; i++) {
        uint32_t piece = 1 << i;
        if (!((player >> i) & 1))
            continue;

        Move m = { piece, piece, 0 };
        stack.push(m);
    }

    // Find all captures and make them longest possible
    Move m;
    uint32_t p, o, a, q;
    while (!stack.is_empty()) {
        m = stack.peek();
        p = player ^ m.start;       // Players pieces (with current one at the start)
        o = opponent ^ m.captured;  // Opponents pieces (without the captured ones)
        a = all_pieces ^ m.start;   // All pieces (with captured ones, but without the current one)
        q = queens;                 // Queens on the board (with current one at the start if a queen)

        // If the piece is not a queen
        if (!(m.start & q)) {
            if (m.end & EVEN_ROWS) {
                if (!(m.end & (FIRST_COLUMN | SECOND_COLUMN))) {
                    if ((m.end << EVEN_UP_LEFT) & o && (m.end << (UP_LEFT_CAPT)) & ~a) {
                        Move t = m;
                        t.end = m.end << (UP_LEFT_CAPT);
                        t.captured |= (m.end << EVEN_UP_LEFT);
                        stack.push(t);
                    }
                    if ((m.end << EVEN_DOWN_LEFT) & o && (m.end << (DOWN_LEFT_CAPT)) & ~a) {
                        Move t = m;
                        t.end = m.end << (DOWN_LEFT_CAPT);
                        t.captured |= (m.end << EVEN_DOWN_LEFT);
                        stack.push(t);
                    }
                }
                if (!(m.end & (S_LAST_COLUMN | LAST_COLUMN))) {
                    if ((m.end << EVEN_UP_RIGHT) & o && (m.end << (UP_RIGHT_CAPT)) & ~a) {
                        Move t = m;
                        t.end = m.end << (UP_RIGHT_CAPT);
                        t.captured |= (m.end << EVEN_UP_RIGHT);
                        stack.push(t);
                    }
                    if ((m.end << EVEN_DOWN_RIGHT) & o && (m.end << (DOWN_RIGHT_CAPT)) & ~a) {
                        Move t = m;
                        t.end = m.end << (DOWN_RIGHT_CAPT);
                        t.captured |= (m.end << EVEN_DOWN_RIGHT);
                        stack.push(t);
                    }
                }
            } else { // Piece is in an odd row
                
            }
            stack.pop();
        } else { // Piece is a queen
            uint32_t position = m.end;

                // Up left
                int i = 0;
                int steps[2] = { EVEN_UP_LEFT, ODD_UP_LEFT };
                if (position & ODD_ROWS) {
                    steps[0] = ODD_UP_LEFT;
                    steps[1] = EVEN_UP_LEFT;
                }
                bool is_capture = false;
                while (!(position & (FIRST_COLUMN | SECOND_COLUMN))) {
                    int step = steps[i++ & 1];
                    position <<= step;

                    if (is_capture) {
                        if (position & a)
                            break;
                        Move t = m;
                        t.end = position;
                        stack.push(t);
                        continue;
                    }

                    if (position & p) {
                        break;
                    }

                    if ((position & o) && !(position << steps[i & 1] & a)) {
                        is_capture = true;

                        break;
                    } else if ((position & o) && (position << steps[i & 1] & a)) {
                        break;
                    }
                }
        }
    }

    // If no captures were found, find all non-captures
}