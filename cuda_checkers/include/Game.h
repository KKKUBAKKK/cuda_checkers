#ifndef GAME_H
#define GAME_H

#include <Player.h>

class Game {
public:
    Player players[2];
    bool is_first_manual;
    bool is_second_manual;

    float time_limit_ms;
    int max_games;
    int max_iterations;

    Game(float time_limit_ms = TIME_LIMIT_MS, int max_games = MAX_GAMES, int max_iterations = MAX_ITERATIONS,
         bool is_first_cpu = true, bool is_second_cpu = true, bool is_first_manual = false, bool is_second_manual = false) :
         time_limit_ms(time_limit_ms), max_games(max_games), max_iterations(max_iterations), is_first_cpu(is_first_cpu), 
         is_second_cpu(is_second_cpu), is_first_manual(is_first_manual), is_second_manual(is_second_manual) {

        if (!is_first_manual)
            players[0] = Player(true, is_first_cpu, max_games, max_iterations, time_limit_ms);

        if (!is_second_manual)
            players[1] = Player(false, is_second_cpu, max_games, max_iterations, time_limit_ms);
    };

    ~Game();
    void run();
};

#endif // GAME_H