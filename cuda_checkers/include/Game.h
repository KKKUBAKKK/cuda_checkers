#ifndef GAME_H
#define GAME_H

#include <Player.h>

class Game {
public:
    Player players[2];
    bool is_first_manual;
    bool is_second_manual;
    bool is_first_cpu;
    bool is_second_cpu;

    float time_limit_ms;
    int max_games;
    int max_iterations;

    Game(float time_limit_ms = TIME_LIMIT_MS, int max_games = MAX_GAMES, int max_iterations = MAX_ITERATIONS,
            bool is_first_cpu = true, bool is_second_cpu = true, bool is_first_manual = false, bool is_second_manual = false);

    ~Game();

    void run();
    Move parse_user_input(Board board);
};

#endif // GAME_H