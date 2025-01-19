#include "../include/Board.h"
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "../include/Move.h"
#include "../include/Stack.h"

#define MAX_MOVES 144

__host__ void Board::print_board() {
    // Print column headers
    std::cout << "   A  B  C  D  E  F  G  H\n";

    // Depending on whose move it is, decide the orientation of the board
    if (!whiteToMove) {
        for (int row = 0; row < 8; ++row) {
            std::cout << 1 + row << ' '; // Print row number
            for (int col = 0; col < 8; ++col) {
                print_square(row, col);
            }
            std::cout << ' ' << 1 + row << '\n'; // Print row number again for easier reading
        }
    } else {
        for (int row = 7; row >= 0; --row) {
            std::cout << 1 + row << ' '; // Print row number
            for (int col = 0; col < 8; ++col) {
                print_square(row, col);
            }
            std::cout << ' ' << 1 + row << '\n'; // Print row number again for easier reading
        }
    }

    // Print column headers
    std::cout << "   A  B  C  D  E  F  G  H\n";
}

__host__ void Board::print_square(int row, int col) {
    if (row % 2 != col % 2) {
        std::cout << ".  "; // Empty square
        return;
    }

    int index = row * 4 + (col / 2);
    bool is_white_piece = (white >> index) & 1;
    bool is_black_piece = (black >> index) & 1;
    bool is_queen = (queens >> index) & 1;

    if (is_white_piece) {
        if (is_queen) {
            std::cout << "WQ "; // White Queen
        } else {
            std::cout << "W  "; // White piece
        }
    } else if (is_black_piece) {
        if (is_queen) {
            std::cout << "BQ "; // Black Queen
        } else {
            std::cout << "B  "; // Black piece
        }
    } else {
        std::cout << ".  "; // Empty square
    }
}

// TODO: Add randomness and time limit
__host__ __device__ int Board::simulate_n_games(int n) {
    int score = 0;
    for (int i = 0; i < n; i++) {
        score += simulate_game();
    }

    return score;
}

// TODO: Add randomness
__host__ __device__ int Board::simulate_game() {
    Board board = this;
    Move moves[MAX_MOVES];

    while (true) {
        int num_moves = board.generate_moves(moves);
        if (num_moves == 0) {
            return 0;
        }

        // Pick random number between 0 and num_moves
        int random = 0;

        // Apply the random picked move
        board = board.apply_move(moves[random]);
        
        if (!board.white) {
            return whiteToMove ? -1 : 1;
        }

        if (!board.black) {
            return whiteToMove ? 1 : -1;
        }

    }
}

__host__ __device__ Board Board::apply_move(const Move &move) {
    Board new_board = this;
    if (new_board.whiteToMove) {
        new_board.white ^= move.start | move.end;
        new_board.black ^= move.captured;
    } else {
        new_board.black ^= move.start | move.end;
        new_board.white ^= move.captured;
    }
    if (new_board.queens & move.start) {
        new_board.queens ^= move.start;
        new_board.queens |= move.end;
    } else {
        new_board.queens |= move.end & LAST_ROW;
    }
    new_board.queens &= new_board.white | new_board.black;

    new_board.whiteToMove = !new_board.whiteToMove;
    return new_board;
}

__host__ __device__ int Board::generate_moves(Move *moves) {
	// Move moves[MAX_MOVES];
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
                if (((m.end >> left[1]) & o) && ((m.end >> (DOWN_LEFT_CAPT)) & ~a)) {
                    Move t = m;
                    t.end = m.end >> (DOWN_LEFT_CAPT);
                    t.captured |= (m.end >> left[1]);
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
                if (((m.end >> right[1]) & o) && ((m.end >> (DOWN_RIGHT_CAPT)) & ~a)) {
                    Move t = m;
                    t.end = m.end >> (DOWN_RIGHT_CAPT);
                    t.captured |= (m.end >> right[1]);
                    stack.push(t);
                }
            }
        } else { // Piece is a queen
            uint32_t position = m.end;

            // Initialize steps and constraints for looping through the directions
            int steps[8];
            if (position & EVEN_ROWS) {
                steps[0] = EVEN_UP_LEFT;    steps[1] = ODD_UP_LEFT;
                steps[2] = EVEN_UP_RIGHT;   steps[3] = ODD_UP_RIGHT;
                steps[4] = -EVEN_DOWN_LEFT;  steps[5] = -ODD_DOWN_LEFT;
                steps[6] = -EVEN_DOWN_RIGHT; steps[7] = -ODD_DOWN_RIGHT;
            } else {
                steps[0] = ODD_UP_LEFT;     steps[1] = EVEN_UP_LEFT;
                steps[2] = ODD_UP_RIGHT;    steps[3] = EVEN_UP_RIGHT;
                steps[4] = -ODD_DOWN_LEFT;   steps[5] = -EVEN_DOWN_LEFT;
                steps[6] = -ODD_DOWN_RIGHT;  steps[7] = -EVEN_DOWN_RIGHT;
            }

            int constraints[4];
            constraints[0] = FIRST_COLUMN; constraints[1] = SECOND_COLUMN;
            constraints[2] = LAST_COLUMN;  constraints[3] = S_LAST_COLUMN;

            for (int i = 0; i < 4; i++) {
                int j = 0;
                bool is_capture = false;
                Move tm = m;

                while (!(position & constraints[(i & 1) * 2])) {
                    if ((position & constraints[(i & 1) * 2 + 1]) && !is_capture)
                        break;

                    int step = steps[(i * 2) + (j++ & 1)];
                    if (step < 0) {
                        position >>= -step;
                    } else {
                        position <<= step;
                    }
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
        return num_moves;
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
                if ((m.end >> left[1]) & ~a) {
                    Move t = m;
                    t.end = m.end >> (left[1]);
                    moves[num_moves++] = t;
                }
            }
            if (!(m.end & LAST_COLUMN)) {
                if ((m.end << right[0]) & ~a) {
                    Move t = m;
                    t.end = m.end << (right[0]);
                    moves[num_moves++] = t;
                }
                if ((m.end >> right[1]) & ~a) {
                    Move t = m;
                    t.end = m.end >> (right[1]);
                    moves[num_moves++] = t;
                }
            }
        } else { // Piece is a queen
            uint32_t position = m.end;

            // Initialize steps and constraints for looping through the directions
            int steps[8];
            if (position & EVEN_ROWS) {
                steps[0] = EVEN_UP_LEFT;    steps[1] = ODD_UP_LEFT;
                steps[2] = EVEN_UP_RIGHT;   steps[3] = ODD_UP_RIGHT;
                steps[4] = -EVEN_DOWN_LEFT;  steps[5] = -ODD_DOWN_LEFT;
                steps[6] = -EVEN_DOWN_RIGHT; steps[7] = -ODD_DOWN_RIGHT;
            } else {
                steps[0] = ODD_UP_LEFT;     steps[1] = EVEN_UP_LEFT;
                steps[2] = ODD_UP_RIGHT;    steps[3] = EVEN_UP_RIGHT;
                steps[4] = -ODD_DOWN_LEFT;   steps[5] = -EVEN_DOWN_LEFT;
                steps[6] = -ODD_DOWN_RIGHT;  steps[7] = -EVEN_DOWN_RIGHT;
            }

            int constraints[2];
            constraints[0] = FIRST_COLUMN;
            constraints[1] = LAST_COLUMN;

            for (int i = 0; i < 4; i++) {
                int j = 0;
                Move tm = m;

                while (!(position & constraints[i & 1])) {
                    int step = steps[i * 2 + (j++ & 1)];
                    if (step < 0)
                        position >>= -step;
                    else
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
    return num_moves;
}