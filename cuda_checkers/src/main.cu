#include <iostream>
#include "../include/Board.h" // Include the header file where Board is defined

int main() {
    Board white;
    white.print_board();

    Move moves[32];
    int n = white.generate_moves(moves);
    for (int i = 0; i < n; i++) {
        Board b = white.apply_move(moves[i]);
        std::cout << "White: " << b.white << ", Black: " << b.black << ", Queens: " << b.queens << std::endl;
        std::cout << "Move " << i + 1 << "  " << std::endl;
        b.print_board();
    }
}