#ifndef NODE_H
#define NODE_H

#include "Board.h"

class Node {
public:
    Board board;
    Node *parent;
    std::vector<Node*> children;
    int score;
    int visits;

    explicit Node(Board board = Board(), Node *parent = nullptr) : board(board), parent(parent), children(), score(0), visits(0) {}
    Node(Node *parent, bool whiteToMove = true) : board(Board(whiteToMove)), parent(nullptr), children(), score(0), visits(0) {}
    
    ~Node() {
        for (Node *child : children) {
            delete child;
        }
    }
};

#endif // NODE_H