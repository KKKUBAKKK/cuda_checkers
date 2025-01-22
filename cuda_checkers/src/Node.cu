#include "Node.h"

Node::Node(Board board, Node *parent) : board(board), parent(parent), children(), score(0), visits(0) {
    Move *moves = new Move[MAX_MOVES];
    Move *stack = new Move[MAX_MOVES];

    int num_moves = this->board.generate_moves(moves, stack);
    for (int i = 0; i < num_moves; i++) {
        possible_moves.push(moves[i]);
    }

    delete[] moves;
    delete[] stack;
}

Node::Node(Node *parent, bool whiteToMove) : board(Board(whiteToMove)), parent(nullptr), children(), score(0), visits(0) {
    Move *moves = new Move[MAX_MOVES];
    Move *stack = new Move[MAX_MOVES];

    int num_moves = this->board.generate_moves(moves, stack);
    for (int i = 0; i < num_moves; i++) {
        possible_moves.push(moves[i]);
    }

    delete[] moves;
    delete[] stack;
}

Node::~Node() {
    for (Node *child : children) {
        delete child;
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
    return board.white == 0 || board.black == 0;
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