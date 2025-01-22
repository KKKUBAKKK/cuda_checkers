#include "Game.h"
#include <iostream>

void Game::run() {
    Board board;
    bool game_over = false;
    int turn = 0;

    std::cout << "Starting game..." << std::endl;

    while (!game_over) {
        board.print_board();

        if ((turn == 0 && is_first_manual) || (turn == 1 && is_second_manual)) {
            // Manual move
            // std::cout << "Player " << (turn + 1) << " turn:" << std::endl;
            // int from_row, from_col, to_row, to_col;
            // std::cout << "Enter move (from_row from_col to_row to_col): ";
            // std::cin >> from_row >> from_col >> to_row >> to_col;
            // Move move = {from_row, from_col, to_row, to_col};
            // board = board.apply_move(move);
        } else {
            // NPC move
            std::cout << "Player " << (turn + 1) << " turn:\n";
            board = players[turn].make_move(board);
        }

        // TODO: add draw condition and win if no moves available
        // Check for game end conditions
        if (board.white == 0 || board.black == 0) {
            game_over = true;
            std::cout << "Game over! ";
            if (board.white == 0) {
                std::cout << "Black wins!\n";
            } else {
                std::cout << "White wins!\n";
            }
        }

        // Switch turn
        turn = 1 - turn;
    }
}