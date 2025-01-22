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
    std::queue<Move> possible_moves;

    explicit Node(Board board = Board(), Node *parent = nullptr);
    Node(Node *parent, bool whiteToMove = true);
    ~Node();

    Move get_move() const;
    bool is_expanded() const;
    bool is_end() const;
    float get_UCT_value() const;
};

#endif // NODE_H