#ifndef NODE_H
#define NODE_H

#include "Board.h"
#include "Move.h"
#include <vector>
#include <queue>
#include <cassert>

class Node {
public:
    Board board;
    Node *parent;
    std::vector<Node*> children;
    float score;
    float visits;
    int white_queen_moves;
    int black_queen_moves;
    std::queue<Move> possible_moves;

    explicit Node(Board board = Board(), Node *parent = nullptr);
    Node(Node *parent, bool whiteToMove = true);
    ~Node();

    Move get_move();
    bool is_expanded() const;
    bool is_end() const;
    float get_UCT_value() const;
    int white_score() const;
};

#endif // NODE_H