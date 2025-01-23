#include "Game.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <regex>
#include "Board.h"
#include "Move.h"

static Game Game::getGameInfo() {
    float time_limit_ms = TIME_LIMIT_MS;
    int max_games = MAX_GAMES;
    int max_iterations = MAX_ITERATIONS;
    bool is_first_cpu = true;
    bool is_second_cpu = true;
    bool is_first_manual = false;
    bool is_second_manual = false;

    std::cout << "Enter time limit for each move in milliseconds (default: 1000): ";
    std::cin >> time_limit_ms;

    std::cout << "Enter maximum number of games to simulate (default: 1000): ";
    std::cin >> max_games;

    std::cout << "Enter maximum number of iterations tree expansion (default: 1000): ";
    std::cin >> max_iterations;

    std::cout << "Is the first player manual? (y/n, default: n): ";
    char is_first_manual_input;
    std::cin >> is_first_manual_input;
    is_first_manual = is_first_manual_input == 'y' || is_first_manual_input == 'Y';

    if (!is_first_manual) {
        std::cout << "Is the first player a CPU? (y/n, default: y): ";
        char is_first_cpu_input;
        std::cin >> is_first_cpu_input;
        is_first_cpu = is_first_cpu_input == 'y' || is_first_cpu_input == 'Y';
    }

    std::cout << "Is the second player manual? (y/n, default: n): ";
    char is_second_manual_input;
    std::cin >> is_second_manual_input;
    is_second_manual = is_second_manual_input == 'y' || is_second_manual_input == 'Y';

    if (!is_second_manual) {
        std::cout << "Is the second player a CPU? (y/n, default: y): ";
        char is_second_cpu_input;
        std::cin >> is_second_cpu_input;
        is_second_cpu = is_second_cpu_input == 'y' || is_second_cpu_input == 'Y';
    }

    return Game(time_limit_ms, max_games, max_iterations, is_first_cpu, is_second_cpu, is_first_manual, is_second_manual);
}

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
    Board temp = board;
    bool game_over = false;
    int turn = 0;
    int white_queen_turns = 0;
    int black_queen_turns = 0;

    std::cout << "Starting game..." << std::endl;

    while (!game_over) {
        board.print_board();
        temp = board;

        if ((turn == 0 && is_first_manual) || (turn == 1 && is_second_manual)) {
            // Manual move
            std::cout << "Player " << (turn + 1) << " turn:" << std::endl;
            Move move = parse_user_input(board);
            board = board.apply_move(move);
        } else {
            // NPC move
            std::cout << "Player " << (turn + 1) << " turn:\n";
            board = players[turn].make_move(board);
        }

        // Win no moves: Check for no available moves compared to prev
        if (board.white == temp.white && board.black == temp.black && board.queens == temp.queens) {
            game_over = true;
            std::cout << "Game over! No moves available! ";
            if (temp.whiteToMove) {
                std::cout << "Black wins!" << std::endl;
            } else {
                std::cout << "White wins!" << std::endl;
            }
            break;
        }

        // Check for game end conditions
        if (board.white == 0 || board.black == 0) {
            game_over = true;
            std::cout << "Game over! No pawns left! ";
            if (board.white == 0) {
                std::cout << "Black wins!\n";
            } else {
                std::cout << "White wins!\n";
            }
            break;
        }

        // Update draw condition counters
        uint32_t temp_queens = temp.whiteToMove ? temp.queens & temp.white : temp.queens & temp.black;
        uint32_t board_queens = temp.whiteToMove ? board.queens & board.white : board.queens & board.black;
        if (count_set_bits(temp.white | temp.black) != count_set_bits(board.white | board.black)) {
            if (temp.whiteToMove) {
                white_queen_turns = 0;
            } else {
                black_queen_turns = 0;
            }
        } else if (count_set_bits(temp_queens) == count_set_bits(board_queens) && temp_queens != board_queens) {
            if (temp.whiteToMove) {
                white_queen_turns++;
            } else {
                black_queen_turns++;
            }
        }

        // Check for draw
        if (white_queen_turns >= 15 && black_queen_turns >= 15) {
            game_over = true;
            std::cout << "Game over! Draw!\n";
            break;
        }

        // Switch turn
        turn = 1 - turn;
    }
    board.print_board();
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

        // Get bitmasks for start and end positions
        uint32_t start = coordinates_to_position(from_col, from_row);
        uint32_t end = coordinates_to_position(to_col, to_row);
        if (!start || !end) {
            std::cerr << "Invalid move. Please try again." << std::endl;
            return parse_user_input(board); // Recursively ask for input again
        }

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
        if (!(board.queens & start) && abs(from_row - to_row) != 1 || abs(from_col - to_col) != 1) {
            std::cerr << "Invalid move. Please try again." << std::endl;
            return parse_user_input(board); // Recursively ask for input again
        }

        if (board.queens & start && !are_positions_on_diagonal_empty(from_col, from_row, to_col, to_row, current_player, opponent)) {
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

        // Get bitmasks for start and end positions
        uint32_t start = coordinates_to_position(from_col, from_row);
        uint32_t end = coordinates_to_position(to_col, to_row);
        if (!start || !end) {
            std::cerr << "Invalid move. Please try again." << std::endl;
            return parse_user_input(board); // Recursively ask for input again
        }

        // Set current player and opponent bitmasks
        uint32_t current_player = board.whiteToMove ? board.white : board.black;
        uint32_t opponent = board.whiteToMove ? board.black : board.white;

        // Check if start position is occupied by current player
        if (!(start & current_player)) {
            std::cerr << "Invalid move. Please try again." << std::endl;
            return parse_user_input(board); // Recursively ask for input again
        }

        Board temp = board;
        if (temp.whiteToMove) {
            temp.white ^= start;
        } else {
            temp.black ^= start;
        }
        Move move = validate_single_capture(from_col, from_row, to_col, to_row, temp);
        if (!move.start) {
            std::cerr << "Invalid move. Please try again." << std::endl;
            return parse_user_input(board); // Recursively ask for input again
        }

        return move;
    }
    
    if (std::regex_match(input, multi_capture_regex)) {
        // Multiple captures
        std::vector<std::string> positions;
        std::stringstream ss(input);
        std::string position;

        while (std::getline(ss, position, ':')) {
            positions.push_back(position);
        }

        char from_col = positions[0][0];
        char from_row = positions[0][1];
        char to_col = positions.back()[0];
        char to_row = positions.back()[1];

        uint32_t res_start = coordinates_to_position(from_col, from_row);
        uint32_t res_end = coordinates_to_position(to_col, to_row);
        if (!res_start || !res_end) {
            std::cerr << "Invalid move. Please try again." << std::endl;
            return parse_user_input(board); // Recursively ask for input again
        }

        // Set current player and opponent bitmasks
        uint32_t current_player = board.whiteToMove ? board.white : board.black;
        uint32_t opponent = board.whiteToMove ? board.black : board.white;

        // Check if start position is occupied by current player
        if (!(res_start & current_player)) {
            std::cerr << "Invalid move. Please try again." << std::endl;
            return parse_user_input(board); // Recursively ask for input again
        }

        // Calculate captured pieces positions
        uint32_t captured = 0;
        Board temp = board;
        if (temp.whiteToMove) {
            temp.white ^= res_start;
        } else {
            temp.black ^= res_start;
        }
        for (size_t i = 1; i < positions.size(); ++i) {
            from_col = positions[i - 1][0];
            from_row = positions[i - 1][1];
            to_col = positions[i][0];
            to_row = positions[i][1];

            Move move = validate_single_capture(from_col, from_row, to_col, to_row, temp);
            if (!move.start) {
                std::cerr << "Invalid move. Please try again." << std::endl;
                return parse_user_input(board); // Recursively ask for input again
            }

            captured |= move.captured;

            uint32_t start = coordinates_to_position(from_col, from_row);
            uint32_t end = coordinates_to_position(to_col, to_row);
            if (start & temp.queens) {
                temp.queens ^= start;
                temp.queens |= end;
            }
        }

        return Move{res_start, res_end, captured};
    }

    std::cerr << "Invalid move format. Please try again." << std::endl;
    return parse_user_input(board); // Recursively ask for input again
}

Move Game::validate_single_capture(char from_col, char from_row, char to_col, char to_row, Board board) {
    // Get bitmasks for start and end positions
    uint32_t start = coordinates_to_position(from_col, from_row);
    uint32_t end = coordinates_to_position(to_col, to_row);
    if (!start || !end) {
        return Move{0, 0, 0};
    }

    // Check if piece is a queen, if not check if move is valid
    if (!(board.queens & start) && (abs(from_col - to_col) != 1 || abs(from_row - to_row) != 1)) {
        return Move{0, 0, 0};
    }

    // Set current player and opponent bitmasks
    uint32_t current_player = board.whiteToMove ? board.white : board.black;
    uint32_t opponent = board.whiteToMove ? board.black : board.white;

    // Check if end position is empty
    if (end & (current_player | opponent)) {
        return Move{0, 0, 0};
    }

    // Find the captured piece
    uint32_t temp = start;
    uint32_t captured = 0;
    while(temp ^ end) {
        if (start < end) {
            temp <<= 1;
        } else {
            temp >>= 1;
        }

        if (temp & (current_player ^ start)) {
            return Move{0, 0, 0};
        }

        if (temp & opponent) {
            // More than one captured piece
            if (captured)   {
                return Move{0, 0, 0};
            }

            captured = temp;
        }
    }

    // No captured pieces
    if (!captured) {
        return Move{0, 0, 0};
    }

    // Check if start, end and captured are on the same diagonal
    std::pair<char, char> capture_coords = position_to_coordinates(captured);
    if (!are_on_same_diagonal(from_col, from_row, to_col, to_row) ||
        !are_on_same_diagonal(from_col, from_row, capture_coords.first, capture_coords.second) ||
        !are_on_same_diagonal(to_col, to_row, capture_coords.first, capture_coords.second)) {
        return Move{0, 0, 0};
    }

    return Move{start, end, captured};
}

// Function to check if two fields are on the same diagonal
bool Game::are_on_same_diagonal(char from_col, char from_row, char to_col, char to_row) {
    // Convert columns 'a'-'h' to 0-7
    int from_col_idx = from_col - 'a';
    int to_col_idx = to_col - 'a';

    // Convert rows '1'-'8' to 0-7
    int from_row_idx = from_row - '1';
    int to_row_idx = to_row - '1';

    // Check if the absolute difference between columns is equal to the absolute difference between rows
    return std::abs(from_col_idx - to_col_idx) == std::abs(from_row_idx - to_row_idx);
}

// Function to convert a uint32_t position to char coordinates
std::pair<char, char> Game::position_to_coordinates(uint32_t position) {
    // Find the bit position (0-based index)
    int bit_position = 0;
    while (position >>= 1) {
        bit_position++;
    }

    // Calculate row and column indices
    int row_idx = bit_position / 4;
    int col_idx = (bit_position % 4) * 2 + (row_idx % 2);

    // Convert indices to characters
    char row = '1' + row_idx;
    char col = 'a' + col_idx;

    return {col, row};
}

uint32_t Game::coordinates_to_position(char col, char row) {
    if (((row - '1') % 2 != (col - 'a') % 2)) {
        return 0;
    }

    // Convert columns 'a'-'h' to 0-7
    int col_idx = col - 'a';

    // Convert rows '1'-'8' to 0-7
    int row_idx = row - '1';

    // Calculate the bit position (0-based index)
    int bit_position = row_idx * 4 + col_idx / 2;

    return 1 << bit_position;
}

// Function to check if all positions on the diagonal between two places are empty
bool Game::are_positions_on_diagonal_empty(char from_col, char from_row, char to_col, char to_row, uint32_t current_player, uint32_t opponent) {
    int from_col_idx = from_col - 'a';
    int from_row_idx = from_row - '1';
    int to_col_idx = to_col - 'a';
    int to_row_idx = to_row - '1';

    int col_step = (to_col_idx > from_col_idx) ? 1 : -1;
    int row_step = (to_row_idx > from_row_idx) ? 1 : -1;

    int col = from_col_idx + col_step;
    int row = from_row_idx + row_step;

    while (col != to_col_idx && row != to_row_idx) {
        uint32_t position = coordinates_to_position('a' + col, '1' + row);
        if (position & (current_player | opponent)) {
            return false;
        }
        col += col_step;
        row += row_step;
    }

    return true;
}

// Function to count the number of bits set to one in a uint32_t value
int Game::count_set_bits(uint32_t n) {
    int count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}