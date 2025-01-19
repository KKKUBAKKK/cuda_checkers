#include "../include/Board.h"
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "../include/Move.h"
#include "../include/Stack.h"

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
    int stack_size;
    uint32_t p, o, a, q;
    while (!stack.is_empty()) {
        m = stack.pop();
        stack_size = stack.size();
        p = player ^ m.start;       // Players pieces (with current one at the start)
        o = opponent ^ m.captured;  // Opponents pieces (without the captured ones)
        a = all_pieces ^ m.start;   // All pieces (with captured ones, but without the current one)
        q = queens;                 // Queens on the board (with current one at the start if a queen)

        // If the piece is not a queen
        if (!(m.start & q)) {
            int left[2], right[2];
            if (m.end & EVEN_ROWS) {        // Piece is in an even row
                left[0] = EVEN_UP_LEFT;
                left[1] = EVEN_DOWN_LEFT;
                right[0] = EVEN_UP_RIGHT;
                right[1] = EVEN_DOWN_RIGHT;
            } else {                        // Piece is in an odd row
                left[0] = ODD_UP_LEFT;
                left[1] = ODD_DOWN_LEFT;
                right[0] = ODD_UP_RIGHT;
                right[1] = ODD_DOWN_RIGHT;
            }
            if (!(m.end & (FIRST_COLUMN | SECOND_COLUMN))) {
                    if (((m.end << left[0]) & o) && ((m.end << (UP_LEFT_CAPT)) & ~a)) {
                        Move t = m;
                        t.end = m.end << (UP_LEFT_CAPT);
                        t.captured |= (m.end << left[0]);
                        stack.push(t);
                    }
                    if (((m.end << left[1]) & o) && ((m.end << (DOWN_LEFT_CAPT)) & ~a)) {
                        Move t = m;
                        t.end = m.end << (DOWN_LEFT_CAPT);
                        t.captured |= (m.end << left[1]);
                        stack.push(t);
                    }
                }
                if (!(m.end & (S_LAST_COLUMN | LAST_COLUMN))) {
                    if (((m.end << right[0]) & o) && ((m.end << (UP_RIGHT_CAPT)) & ~a)) {
                        Move t = m;
                        t.end = m.end << (UP_RIGHT_CAPT);
                        t.captured |= (m.end << right[0]);
                        stack.push(t);
                    }
                    if (((m.end << right[1]) & o) && ((m.end << (DOWN_RIGHT_CAPT)) & ~a)) {
                        Move t = m;
                        t.end = m.end << (DOWN_RIGHT_CAPT);
                        t.captured |= (m.end << right[1]);
                        stack.push(t);
                    }
                }
        } else { // Piece is a queen
            uint32_t position = m.end;

            // Initialize steps and constraints for looping through the directions
            int steps[4, 2];
            if (position & EVEN_ROWS) {
                steps[0, 0] = EVEN_UP_LEFT;    steps[0, 1] = ODD_UP_LEFT;
                steps[1, 0] = EVEN_UP_RIGHT;   steps[1, 1] = ODD_UP_RIGHT;
                steps[2, 0] = EVEN_DOWN_LEFT;  steps[2, 1] = ODD_DOWN_LEFT;
                steps[3, 0] = EVEN_DOWN_RIGHT; steps[3, 1] = ODD_DOWN_RIGHT;
            } else {
                steps[0, 0] = ODD_UP_LEFT;     steps[0, 1] = EVEN_UP_LEFT;
                steps[1, 0] = ODD_UP_RIGHT;    steps[1, 1] = EVEN_UP_RIGHT;
                steps[2, 0] = ODD_DOWN_LEFT;   steps[2, 1] = EVEN_DOWN_LEFT;
                steps[3, 0] = ODD_DOWN_RIGHT;  steps[3, 1] = EVEN_DOWN_RIGHT;
            }

            int constraints[2,2];
            constraints[0, 0] = FIRST_COLUMN; constraints[0, 1] = SECOND_COLUMN;
            constraints[1, 0] = LAST_COLUMN;  constraints[1, 1] = S_LAST_COLUMN;

            for (int i = 0; i < 4; i++) {
                int j = 0;
                bool is_capture = false;
                Move tm = m;

                while (!(position & constraints[i & 1, 0])) {
                    if ((position & constraints[i & 1, 1]) && !is_capture)
                        break;

                    int step = steps[i, j++ & 1];
                    position <<= step;
                    if (!position)
                        break;

                    if (is_capture) {
                        if (position & a)
                            break;
                        Move t = tm;
                        t.end = position;
                        stack.push(t);
                        continue;
                    }

                    if (position & (p | tm.captured)) {
                        break;
                    }

                    if ((position & o)) {
                        is_capture = true;
                        tm.captured |= position;
                    }
                }
            }
        }

        // If the stack size is the same, no additional captures were found
        if (stack.size() == stack_size && m.captured) {
            moves[num_moves++] = m;
        }
    }

    // If no captures were found, find all non-captures
    if (num_moves) {
        // Somehow return the created captures;
        // TODO: Implement this
        return;
    }

    // Find all pieces
    for (int i = 0; i < BOARD_SIZE; i++) {
        uint32_t piece = 1 << i;
        if (!((player >> i) & 1))
            continue;

        Move m = { piece, piece, 0 };
        stack.push(m);
    }

    // Find all non-captures
    while (!stack.is_empty()) {
        m = stack.pop();
        stack_size = stack.size();
        p = player ^ m.start;       // Players pieces (with current one at the start)
        o = opponent ^ m.captured;  // Opponents pieces (without the captured ones)
        a = all_pieces ^ m.start;   // All pieces (with captured ones, but without the current one)
        q = queens;                 // Queens on the board (with current one at the start if a queen)

        // If the piece is not a queen
        if (!(m.start & q)) {
            int left[2], right[2];
            if (m.end & EVEN_ROWS) {        // Piece is in an even row
                left[0] = EVEN_UP_LEFT;
                left[1] = EVEN_DOWN_LEFT;
                right[0] = EVEN_UP_RIGHT;
                right[1] = EVEN_DOWN_RIGHT;
            } else {                        // Piece is in an odd row
                left[0] = ODD_UP_LEFT;
                left[1] = ODD_DOWN_LEFT;
                right[0] = ODD_UP_RIGHT;
                right[1] = ODD_DOWN_RIGHT;
            }
            if (!(m.end & FIRST_COLUMN)) {
                if ((m.end << left[0]) & ~a) {
                    Move t = m;
                    t.end = m.end << (left[0]);
                    moves[num_moves++] = t;
                }
                if ((m.end << left[1]) & ~a) {
                    Move t = m;
                    t.end = m.end << (left[1]);
                    moves[num_moves++] = t;
                }
            }
            if (!(m.end & LAST_COLUMN)) {
                if ((m.end << right[0]) & ~a) {
                    Move t = m;
                    t.end = m.end << (right[0]);
                    moves[num_moves++] = t;
                }
                if ((m.end << right[1]) & ~a) {
                    Move t = m;
                    t.end = m.end << (right[1]);
                    moves[num_moves++] = t;
                }
            }
        } else { // Piece is a queen
            uint32_t position = m.end;

            // Initialize steps and constraints for looping through the directions
            int steps[4, 2];
            if (position & EVEN_ROWS) {
                steps[0, 0] = EVEN_UP_LEFT;    steps[0, 1] = ODD_UP_LEFT;
                steps[1, 0] = EVEN_UP_RIGHT;   steps[1, 1] = ODD_UP_RIGHT;
                steps[2, 0] = EVEN_DOWN_LEFT;  steps[2, 1] = ODD_DOWN_LEFT;
                steps[3, 0] = EVEN_DOWN_RIGHT; steps[3, 1] = ODD_DOWN_RIGHT;
            } else {
                steps[0, 0] = ODD_UP_LEFT;     steps[0, 1] = EVEN_UP_LEFT;
                steps[1, 0] = ODD_UP_RIGHT;    steps[1, 1] = EVEN_UP_RIGHT;
                steps[2, 0] = ODD_DOWN_LEFT;   steps[2, 1] = EVEN_DOWN_LEFT;
                steps[3, 0] = ODD_DOWN_RIGHT;  steps[3, 1] = EVEN_DOWN_RIGHT;
            }

            int constraints[2];
            constraints[0] = FIRST_COLUMN;
            constraints[1] = LAST_COLUMN;

            // Add all valid moves to the moves array, but also push them to the stack
            for (int i = 0; i < 4; i++) {
                int j = 0;
                Move tm = m;

                while (!(position & constraints[i & 1])) {
                    int step = steps[i, j++ & 1];
                    position <<= step;
                    if (!position)
                        break;

                    if (position & a)
                        break;

                    Move t = tm;
                    t.end = position;
                    moves[num_moves++] = t;
                }
            }
        }
    }

    // Somehow return the created non-captures;
    // TODO: Implement this
    // return moves;
    return;
}