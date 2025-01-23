#include "Node.h"
#include "Game.h"
#include <iostream>

Node::Node(Board board, Node *parent) : board(board), parent(parent), children(), score(0), visits(0) {
    Move moves[MAX_MOVES];
    Move stack[MAX_MOVES];

    int num_moves = this->board.generate_moves(moves, stack);
    for (int i = 0; i < num_moves; i++) {
        possible_moves.push(moves[i]);
    }

    if (parent == nullptr) {
        white_queen_moves = 0;
        black_queen_moves = 0;
    } else if (Game::count_set_bits(board.white | board.black) != Game::count_set_bits(parent->board.white | parent->board.black)) {
        if (board.whiteToMove) {
            white_queen_moves = parent->white_queen_moves;
            black_queen_moves = 0;
        } else {
            white_queen_moves = 0;
            black_queen_moves = parent->black_queen_moves;
        }
    } else {
        white_queen_moves = parent->white_queen_moves;
        black_queen_moves = parent->black_queen_moves;

        uint32_t new_queens = board.white | board.black;
        uint32_t parent_queens = parent->board.white | parent->board.black;
        if (Game::count_set_bits(new_queens) == Game::count_set_bits(parent_queens) && new_queens != parent_queens) {
            if (board.whiteToMove) {
                black_queen_moves++;
            } else {
                white_queen_moves++;
            }
        }
    }
}

Node::Node(Node *parent, bool whiteToMove) : board(whiteToMove), parent(nullptr), children(), score(0), visits(0) {
    Move moves[MAX_MOVES];
    Move stack[MAX_MOVES];

    int num_moves = this->board.generate_moves(moves, stack);
    for (int i = 0; i < num_moves; i++) {
        possible_moves.push(moves[i]);
    }

    if (parent == nullptr) {
        white_queen_moves = 0;
        black_queen_moves = 0;
    } else if (Game::count_set_bits(board.white | board.black) != Game::count_set_bits(parent->board.white | parent->board.black)) {
        if (board.whiteToMove) {
            white_queen_moves = parent->white_queen_moves;
            black_queen_moves = 0;
        } else {
            white_queen_moves = 0;
            black_queen_moves = parent->black_queen_moves;
        }
    } else {
        white_queen_moves = parent->white_queen_moves;
        black_queen_moves = parent->black_queen_moves;

        uint32_t new_queens = board.white | board.black;
        uint32_t parent_queens = parent->board.white | parent->board.black;
        if (Game::count_set_bits(new_queens) == Game::count_set_bits(parent_queens) && new_queens != parent_queens) {
            if (board.whiteToMove) {
                black_queen_moves++;
            } else {
                white_queen_moves++;
            }
        }
    }
}

Node::~Node() {
    for (Node *child : children) {
        if (child != nullptr) {
            delete child;
        }
    }
    
    // Find pointer to this node in parent and remove it
    if (parent != nullptr) {
        for (int j = 0; j < parent->children.size(); j++) {
            if (parent->children[j] != nullptr && parent->children[j]->board.is_equal(board)) {
                parent->children[j] = nullptr;
                break;
            }
        }
    }
}

Move Node::get_move() {
    assert (!possible_moves.empty());
    Move move = possible_moves.front();
    possible_moves.pop();
    return move;
}

bool Node::is_expanded() const {
    return possible_moves.empty();
}

bool Node::is_end() const {
    // If >= 15 queen moves wihout captures, the game is over
    if (white_queen_moves >= 15 && black_queen_moves >= 15) {
        return true;
    }

    // If no children (no moves possible), the game is over
    if (possible_moves.empty() && children.empty()) {
        return true;
    }

    // If no white or black pieces, the game is over
    if (board.white == 0) {
        return true;
    }

    if (board.black == 0) {
        return true;
    }
    
    return false;
}

int Node::white_score() const {
    // If >= 15 queen moves wihout captures, the game is over
    if (white_queen_moves >= 15 && black_queen_moves >= 15) return 0.5;

    // If no children (no moves possible), the game is over
    if (children.empty() && possible_moves.empty()) return (board.whiteToMove) ? 0 : 1;

    // If no white or black pieces, the game is over
    if (board.white == 0) return 0;
    if (board.black == 0) return 1;

    return -1;
}

float Node::get_UCT_value() const {
    if (visits == 0) return std::numeric_limits<float>::infinity();

    const float C = 1.41421356237f; // sqrt(2)
    float exploitation = static_cast<float>(score) / visits;
    float exploration = 0.0f;

    if (parent != nullptr) {
        exploration = C * std::sqrt(std::log(parent->visits) / visits);
    }

    return exploitation + exploration;
}