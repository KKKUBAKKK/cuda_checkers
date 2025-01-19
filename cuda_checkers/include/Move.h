#ifndef MOVE_H
#define MOVE_H

#include <cstdint>

struct Move {
    uint32_t start;
    uint32_t end;
    uint32_t captured;
};

#endif // MOVE_H