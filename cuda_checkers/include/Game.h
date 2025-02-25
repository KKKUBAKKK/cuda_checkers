#ifndef GAME_H
#define GAME_H

#include <Player.h>
#include <fstream>

class Game {
public:
    Player *players[2];
    bool is_first_manual;
    bool is_second_manual;
    bool is_first_cpu;
    bool is_second_cpu;
    std::ofstream file;

    float time_limit_ms;
    int max_games_one;
    int max_games_two;

    explicit Game(std::string fileName, float time_limit_ms = TIME_LIMIT_MS, int max_games_one = MAX_GAMES_CPU, int max_games_two = MAX_GAMES_CPU,
            bool is_first_cpu = true, bool is_second_cpu = true, bool is_first_manual = false, bool is_second_manual = false);

    ~Game();

    static Game getGameInfo(std::string fileName);
    void run();
    Move parse_user_input(Board board);
    Move validate_single_capture(char from_col, char from_row, char to_col, char to_row, Board &board);
    bool are_on_same_diagonal(char from_col, char from_row, char to_col, char to_row);
    static std::pair<char, char> position_to_coordinates(uint32_t position);
    static uint32_t coordinates_to_position(char col, char row);
    bool are_positions_on_diagonal_empty(char from_col, char from_row, char to_col, char to_row, uint32_t current_player, uint32_t opponent);
    static int count_set_bits(uint32_t n);
    Move get_move(Board prev, Board next);
    std::string get_move_string(Move move, Board prevBoard);
};

#endif // GAME_H