#include "../include/Board.h"
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "../include/Move.h"
#include "../include/Stack.h"
#include <cassert>
#include <stack>
#include <vector>
#include "../include/Game.h"
#include <random>
#include <chrono>

__host__ void Board::print_board(std::ofstream &file) {
    // Print column headers
    std::cout << "   A  B  C  D  E  F  G  H\n";
    file << "   A  B  C  D  E  F  G  H\n";
    std::cout << "  +--+--+--+--+--+--+--+--+\n";
    file << "  +--+--+--+--+--+--+--+--+\n";

    // Depending on whose move it is, decide the orientation of the board
    // if (!whiteToMove) {
    //     for (int row = 0; row < 8; ++row) {
    //         std::cout << 1 + row << ' '; // Print row number
    //         file << 1 + row << ' '; // Print row number
    //         for (int col = 0; col < 8; ++col) {
    //             std::cout << "|";
    //             file << "|";
    //             print_square(row, col, file);
    //         }
    //         std::cout << "| " << 1 + row << '\n'; // Print row number again for easier reading
    //         file << "| " << 1 + row << '\n'; // Print row number again for easier reading
    //         std::cout << "  +--+--+--+--+--+--+--+--+\n";
    //         file << "  +--+--+--+--+--+--+--+--+\n";
    //     }
    // } else {
        for (int row = 7; row >= 0; --row) {
            std::cout << 1 + row << ' '; // Print row number
            file << 1 + row << ' '; // Print row number
            for (int col = 0; col < 8; ++col) {
                std::cout << "|";
                file << "|";
                print_square(row, col, file);
            }
            std::cout << "| " << 1 + row << '\n'; // Print row number again for easier reading
            file << "| " << 1 + row << '\n'; // Print row number again for easier reading
            std::cout << "  +--+--+--+--+--+--+--+--+\n";
            file << "  +--+--+--+--+--+--+--+--+\n";
        }
    // }

    // Print column headers
    std::cout << "   A  B  C  D  E  F  G  H\n";
    file << "   A  B  C  D  E  F  G  H\n";
}

__host__ void Board::print_square(int row, int col, std::ofstream &file) {
    if (row % 2 != col % 2) {
        std::cout << "  "; // Empty square
        file << "  "; // Empty square
        return;
    }

    int index = row * 4 + (col / 2);
    bool is_white_piece = (white >> index) & 1;
    bool is_black_piece = (black >> index) & 1;
    bool is_queen = (queens >> index) & 1;

    if (is_white_piece) {
        if (is_queen) {
            std::cout << "WQ"; // White Queen
            file << "WQ"; // White Queen
        } else {
            std::cout << "W "; // White piece
            file << "W "; // White piece
        }
    } else if (is_black_piece) {
        if (is_queen) {
            std::cout << "BQ"; // Black Queen
            file << "BQ"; // Black Queen
        } else {
            std::cout << "B "; // Black piece
            file << "B "; // Black piece
        }
    } else {
        std::cout << "  "; // Empty square
        file << "  "; // Empty square
    }
}

__host__ void Board::print_board() {
    // Print column headers
    std::cout << "   A  B  C  D  E  F  G  H\n";
    std::cout << "  +--+--+--+--+--+--+--+--+\n";

    // Depending on whose move it is, decide the orientation of the board
    if (!whiteToMove) {
        for (int row = 0; row < 8; ++row) {
            std::cout << 1 + row << ' '; // Print row number
            for (int col = 0; col < 8; ++col) {
                std::cout << "|";
                print_square(row, col);
            }
            std::cout << "| " << 1 + row << '\n'; // Print row number again for easier reading
            std::cout << "  +--+--+--+--+--+--+--+--+\n";
        }
    } else {
        for (int row = 7; row >= 0; --row) {
            std::cout << 1 + row << ' '; // Print row number
            for (int col = 0; col < 8; ++col) {
                std::cout << "|";
                print_square(row, col);
            }
            std::cout << "| " << 1 + row << '\n'; // Print row number again for easier reading
            std::cout << "  +--+--+--+--+--+--+--+--+\n";
        }
    }

    // Print column headers
    std::cout << "   A  B  C  D  E  F  G  H\n";
}

__host__ void Board::print_square(int row, int col) {
    if (row % 2 != col % 2) {
        std::cout << "  "; // Empty square
        return;
    }

    int index = row * 4 + (col / 2);
    bool is_white_piece = (white >> index) & 1;
    bool is_black_piece = (black >> index) & 1;
    bool is_queen = (queens >> index) & 1;

    if (is_white_piece) {
        if (is_queen) {
            std::cout << "WQ"; // White Queen
        } else {
            std::cout << "W "; // White piece
        }
    } else if (is_black_piece) {
        if (is_queen) {
            std::cout << "BQ"; // Black Queen
        } else {
            std::cout << "B "; // Black piece
        }
    } else {
        std::cout << "  "; // Empty square
    }
}

std::vector<std::pair<Move, std::string>> Board::get_printable_captures() {
    uint32_t player = whiteToMove ? white : black;
    uint32_t opponent = whiteToMove ? black : white;
    uint32_t all_pieces = white | black;

    std::stack<std::pair<Move, std::string>> stack;
    std::vector<std::pair<Move, std::string>> moves;

    // First try to find all captures (forced captures)
    // Find all pieces that can capture
    for (int i = 0; i < BOARD_SIZE; i++) {
        uint32_t piece = 1 << i;
        if (!((player >> i) & 1))
            continue;

        Move m = { piece, piece, 0 };
        auto c = Game::position_to_coordinates(piece);
        std::string s = std::string(1, c.first) + std::string(1, c.second);
        std::pair<Move, std::string> p = { m, s};
        stack.push(p);
    }

    // Find all captures and make them longest possible
    Move m;
    int stack_size;
    uint32_t p, o, a, q;
    while (!stack.empty()) {
        std::pair<Move, std::string> move_pair = stack.top();
        stack.pop();
        m = move_pair.first;
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
                    auto c = Game::position_to_coordinates(t.end);
                    std::string s = move_pair.second + ":" + std::string(1, c.first) + std::string(1, c.second);
                    std::pair<Move, std::string> pt = { t, s };
                    stack.push(pt);
                }
                if (((m.end >> left[1]) & o) && ((m.end >> (DOWN_LEFT_CAPT)) & ~a)) {
                    Move t = m;
                    t.end = m.end >> (DOWN_LEFT_CAPT);
                    t.captured |= (m.end >> left[1]);
                    auto c = Game::position_to_coordinates(t.end);
                    std::string s = move_pair.second + ":" + std::string(1, c.first) + std::string(1, c.second);
                    std::pair<Move, std::string> pt = { t, s };
                    stack.push(pt);
                }
            }
            if (!(m.end & (S_LAST_COLUMN | LAST_COLUMN))) {
                if (((m.end << right[0]) & o) && ((m.end << (UP_RIGHT_CAPT)) & ~a)) {
                    Move t = m;
                    t.end = m.end << (UP_RIGHT_CAPT);
                    t.captured |= (m.end << right[0]);
                    auto c = Game::position_to_coordinates(t.end);
                    std::string s = move_pair.second + ":" + std::string(1, c.first) + std::string(1, c.second);
                    std::pair<Move, std::string> pt = { t, s };
                    stack.push(pt);
                }
                if (((m.end >> right[1]) & o) && ((m.end >> (DOWN_RIGHT_CAPT)) & ~a)) {
                    Move t = m;
                    t.end = m.end >> (DOWN_RIGHT_CAPT);
                    t.captured |= (m.end >> right[1]);
                    auto c = Game::position_to_coordinates(t.end);
                    std::string s = move_pair.second + ":" + std::string(1, c.first) + std::string(1, c.second);
                    std::pair<Move, std::string> pt = { t, s };
                    stack.push(pt);
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
                position = m.end;
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
                        auto c = Game::position_to_coordinates(t.end);
                        std::string s = move_pair.second + ":" + std::string(1, c.first) + std::string(1, c.second);
                        std::pair<Move, std::string> pt = { t, s };
                        stack.push(pt);
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
            moves.push_back(move_pair);
        }
    }

    return moves;
}

__host__ __device__ bool Board::is_equal(Board board) {
    return white == board.white && black == board.black && queens == board.queens && whiteToMove == board.whiteToMove;
}

__device__ float Board::simulate_n_games_gpu(curandState* state, Move *moves, Move *stack, bool is_player_white, int n, float time_limit_ms) {
    float score = 0;
    unsigned long long start = clock64();
    float clock_rate = 1.0f / 1000000.0f;

    for (int i = 0; i < n; i++) {
        if ((clock64() - start) * clock_rate > time_limit_ms) {
            break;
        }
        score += simulate_game_gpu(state, moves, stack, is_player_white);
    }
    return score;
}

__device__ float Board::simulate_game_gpu(curandState* state, Move *moves, Move *stack, bool is_player_white) {
    Board board = *this;
    int black_queen_moves = 0;
    int white_queen_moves = 0;

    while (true) {
        int num_moves = board.generate_moves(moves, stack);
        if (num_moves == 0) {
            if (is_player_white) {
                return board.whiteToMove ? 0 : 1;
            } else {
                return board.whiteToMove ? 1 : 0;
            }
        }
        
        int random = curand(state) % num_moves;

        if (!moves[random].captured && (moves[random].start & board.queens)) {
            if (board.whiteToMove) {
                white_queen_moves++;
            } else {
                black_queen_moves++;
            }
        } else if (moves[random].captured) {
            white_queen_moves = 0;
            black_queen_moves = 0;
        }

        if (white_queen_moves >= 15 && black_queen_moves >= 15) {
            return 0.5;
        }

        board = board.apply_move(moves[random]);
        
        if (!board.white) return is_player_white ? 0 : 1;
        if (!board.black) return is_player_white ? 1 : 0;
    }
}

__host__ float Board::simulate_n_games_cpu(std::mt19937& rng, Move *moves, Move *stack, bool is_player_white, int n, float time_limit_ms) {
    float score = 0;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; i++) {
        auto now = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration<float, std::milli>(now - start).count();
        if (elapsed > time_limit_ms) {
            break;
        }
        score += simulate_game_cpu(rng, moves, stack, is_player_white);
    }
    return score;
}

__host__ float Board::simulate_game_cpu(std::mt19937& rng, Move *moves, Move *stack, bool is_player_white) {
    Board board = *this;
    int black_queen_moves = 0;
    int white_queen_moves = 0;

    while (true) {
        int num_moves = board.generate_moves(moves, stack);
        if (num_moves == 0) {
            if (is_player_white) {
                return board.whiteToMove ? 0 : 1;
            } else {
                return board.whiteToMove ? 1 : 0;
            }
        }
        
        int random = std::uniform_int_distribution<>(0, num_moves-1)(rng);

        if (!moves[random].captured && (moves[random].start & board.queens)) {
            if (board.whiteToMove) {
                white_queen_moves++;
            } else {
                black_queen_moves++;
            }
        } else if (moves[random].captured) {
            white_queen_moves = 0;
            black_queen_moves = 0;
        }

        if (white_queen_moves >= 15 && black_queen_moves >= 15) {
            return 0.5;
        }

        board = board.apply_move(moves[random]);
        
        if (!board.white) return is_player_white ? 0 : 1;
        if (!board.black) return is_player_white ? 1 : 0;
    }
}

__host__ __device__ Board Board::apply_move(const Move &move) {
    Board new_board = *this;
    if (new_board.whiteToMove) {
        new_board.white ^= move.start | move.end;
        new_board.black ^= move.captured;

        if (new_board.queens & move.start) {
            new_board.queens &= new_board.white | new_board.black;
            new_board.queens |= move.end;
        } else {
            new_board.queens |= move.end & LAST_ROW;
        }
    } else {
        new_board.black ^= move.start | move.end;
        new_board.white ^= move.captured;

        if (new_board.queens & move.start) {
            new_board.queens &= new_board.white | new_board.black;
            new_board.queens |= move.end;
        } else {
            new_board.queens |= move.end & FIRST_ROW;
        }
    }
    new_board.queens &= new_board.white | new_board.black;

    new_board.whiteToMove = !new_board.whiteToMove;
    return new_board;
}

__host__ __device__ int Board::generate_moves(Move *movesmem, Move *stackmem) {
    uint32_t player = whiteToMove ? white : black;
    uint32_t opponent = whiteToMove ? black : white;
    uint32_t all_pieces = white | black;

    Stack stack(stackmem, MAX_MOVES);
    Stack moves(movesmem, MAX_MOVES);

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
                position = m.end;
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
            moves.push(m);
        }
    }

    // If no captures were found, find all non-captures
    if (!moves.is_empty()) {
        return moves.size();
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
                if (((m.end << left[0]) & ~a) && whiteToMove) {
                    Move t = m;
                    t.end = m.end << (left[0]);
                    moves.push(t);
                }
                if (((m.end >> left[1]) & ~a) && !whiteToMove) {
                    Move t = m;
                    t.end = m.end >> (left[1]);
                    moves.push(t);
                }
            }
            if (!(m.end & LAST_COLUMN)) {
                if (((m.end << right[0]) & ~a) && whiteToMove) {
                    Move t = m;
                    t.end = m.end << (right[0]);
                    moves.push(t);
                }
                if (((m.end >> right[1]) & ~a) && !whiteToMove) {
                    Move t = m;
                    t.end = m.end >> (right[1]);
                    moves.push(t);
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
                position = m.end;

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
                    moves.push(t);
                }
            }
        }
    }

    // Somehow return the created non-captures;
    return moves.size();
}