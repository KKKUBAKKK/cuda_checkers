#include <iostream>
#include "../include/Board.h" // Include the header file where Board is defined
#include <thread>
#include <chrono>

int main() {
    Board white;
    white.print_board();

    // Move moves[32];
    // int n = white.generate_moves(moves);
    // for (int i = 0; i < n; i++) {
    //     Board b = white.apply_move(moves[i]);
    //     std::cout << "White: " << b.white << ", Black: " << b.black << ", Queens: " << b.queens << std::endl;
    //     std::cout << "Move " << i + 1 << "  " << std::endl;
    //     b.print_board();
    // }

    Board board = white;
    Move moves[144];

    while (true) {
        int num_moves = board.generate_moves(moves);
        if (num_moves == 0) {
            if (board.whiteToMove) {
                std::cout << "Black wins, no moves!" << std::endl;
                break;
            } else {
                std::cout << "White wins, no moves!" << std::endl;
                break;
            }
        }

        // Pick random number between 0 and num_moves
        int random = 0;

        // Apply the random picked move
        board = board.apply_move(moves[random]);
        if (board.whiteToMove) {
            std::cout << "Black moved, White to move now: " << std::endl;
        } else {
            std::cout << "White moved, Black to move now: " << std::endl;
        }
        board.print_board();

        // std::this_thread::sleep_for(std::chrono::seconds(1));

        if (!board.white) {
            std::cout << "Black wins, all captured!" << std::endl;
            break;
        }

        if (!board.black) {
            std::cout << "White wins, all captured!" << std::endl;
            break;
        }
    }
}