#include "Game.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <regex>
#include "Board.h"
#include "Move.h"

Game::Game(float time_limit_ms, int max_games, int max_iterations, bool is_first_cpu, 
           bool is_second_cpu, bool is_first_manual, bool is_second_manual) :
    time_limit_ms(time_limit_ms), max_games(max_games), max_iterations(max_iterations), is_first_cpu(is_first_cpu), 
    is_second_cpu(is_second_cpu), is_first_manual(is_first_manual), is_second_manual(is_second_manual) {

    if (!is_first_manual)
    players[0] = Player(true, is_first_cpu, max_games, max_iterations, time_limit_ms);

    if (!is_second_manual)
    players[1] = Player(false, is_second_cpu, max_games, max_iterations, time_limit_ms);
};

Game::~Game() {
    // delete players[0];
    // delete players[1];
}

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

Move Game::parse_user_input(Board board) {
    std::string input;
    std::cout << "Enter your move: ";
    std::getline(std::cin, input);

    // Regular expressions for different move types
    std::regex move_regex("^[a-h][1-8]-[a-h][1-8]$");
    std::regex capture_regex("^[a-h][1-8]:[a-h][1-8]$");
    std::regex multi_capture_regex("^[a-h][1-8](:[a-h][1-8])+$");

    if (std::regex_match(input, move_regex)) {
        // Simple move
        char from_col = input[0];
        char from_row = input[1];
        char to_col = input[3];
        char to_row = input[4];

        // Check if move uses black squares only
        if (((from_row - '1') % 2 != (from_col - 'a') % 2) ||
            ((to_row - '1') % 2 != (to_col - 'a') % 2)) {
            std::cerr << "Invalid move. Please try again." << std::endl;
            return parse_user_input(board); // Recursively ask for input again
        }

        // Get bitmasks for start and end positions
        uint32_t start = 1 << ((from_row - '1') * 4 + (from_col - 'a') / 2);
        uint32_t end = 1 << ((to_row - '1') * 4 + (to_col - 'a') / 2);

        // Set current player and opponent bitmasks
        uint32_t current_player = board.whiteToMove ? board.white : board.black;
        uint32_t opponent = board.whiteToMove ? board.black : board.white;

        // Check if start position is occupied by current player
        if (!(start & current_player)) {
            std::cerr << "Invalid move. Please try again." << std::endl;
            return parse_user_input(board); // Recursively ask for input again
        }

        // Check if end position is empty
        if (end & (current_player | opponent)) {
            std::cerr << "Invalid move. Please try again." << std::endl;
            return parse_user_input(board); // Recursively ask for input again
        }

        // Check if move doesn't capture any pieces
        if (abs(from_row - to_row) != 1 || abs(from_col - to_col) != 1) {
            std::cerr << "Invalid move. Please try again." << std::endl;
            return parse_user_input(board); // Recursively ask for input again
        }

        return Move{start, end, 0};
    }
    
    if (std::regex_match(input, capture_regex)) {
        // Single capture
        char from_col = input[0];
        char from_row = input[1];
        char to_col = input[3];
        char to_row = input[4];

        // Check if move uses black squares only
        if (((from_row - '1') % 2 != (from_col - 'a') % 2) ||
            ((to_row - '1') % 2 != (to_col - 'a') % 2)) {
            std::cerr << "Invalid move. Please try again." << std::endl;
            return parse_user_input(board); // Recursively ask for input again
        }

        // Get bitmasks for start and end positions
        uint32_t start = 1 << ((from_row - '1') * 4 + (from_col - 'a') / 2);
        uint32_t end = 1 << ((to_row - '1') * 4 + (to_col - 'a') / 2);

        // Calculate captured piece bitmask
        uint32_t captured = 1 << (((from_row - '1' + to_row - '1') / 2) * 4 + ((from_col - 'a' + to_col - 'a') / 2) / 2);

        // Set current player and opponent bitmasks
        uint32_t current_player = board.whiteToMove ? board.white : board.black;
        uint32_t opponent = board.whiteToMove ? board.black : board.white;

        // Check if start position is occupied by current player
        if (!(start & current_player)) {
            std::cerr << "Invalid move. Please try again." << std::endl;
            return parse_user_input(board); // Recursively ask for input again
        }

        // Check if end position is empty
        if (end & (current_player | opponent)) {
            std::cerr << "Invalid move. Please try again." << std::endl;
            return parse_user_input(board); // Recursively ask for input again
        }

        // Check if captured piece is opponent's piece
        if (!(captured & opponent)) {
            std::cerr << "Invalid move. Please try again." << std::endl;
            return parse_user_input(board); // Recursively ask for input again
        }

        // Check if move captures exactly one piece
        if (abs(from_row - to_row) != 2 || abs(from_col - to_col) != 2) {
            std::cerr << "Invalid move. Please try again." << std::endl;
            return parse_user_input(board); // Recursively ask for input again
        }

        return Move{start, end, captured};
    }
    
    // TODO: Implement multi-capture move (check if it's valid)
    if (std::regex_match(input, multi_capture_regex)) {
        // Multiple captures
        std::vector<std::string> positions;
        std::stringstream ss(input);
        std::string position;

        while (std::getline(ss, position, ':')) {
            positions.push_back(position);
        }

        uint32_t start = 1 << ((positions[0][1] - '1') * 4 + (positions[0][0] - 'a') / 2);
        uint32_t end = 1 << ((positions.back()[1] - '1') * 4 + (positions.back()[0] - 'a') / 2);

        // Calculate captured pieces positions
        uint32_t captured = 0;
        for (size_t i = 1; i < positions.size(); ++i) {
            uint32_t intermediate = ((positions[i - 1][1] - '1' + positions[i][1] - '1') / 2) * 4 + ((positions[i - 1][0] - 'a' + positions[i][0] - 'a') / 2) / 2;
            captured |= (1 << intermediate);
        }

        return Move{start, end, captured};
    }

    std::cerr << "Invalid move format. Please try again." << std::endl;
    return parse_user_input(board); // Recursively ask for input again
}