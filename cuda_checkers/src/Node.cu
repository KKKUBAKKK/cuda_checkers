#include "Node.h"

Node::Node(Board board = Board(), Node *parent = nullptr) : board(board), parent(parent), children(), score(0), visits(0) {
    Move *moves = new Move[MAX_MOVES];
    Move *stack = new Move[MAX_MOVES];

    int num_moves = this->board.generateMoves(moves, stack);
    for (int i = 0; i < num_moves; i++) {
        possible_moves.push(moves[i]);
    }

    delete[] moves;
    delete[] stack;
}

Node::Node(Node *parent, bool whiteToMove = true) : board(Board(whiteToMove)), parent(nullptr), children(), score(0), visits(0) {
    Move *moves = new Move[MAX_MOVES];
    Move *stack = new Move[MAX_MOVES];

    int num_moves = this->board.generateMoves(moves, stack);
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

Move Node::get_move() const {
    assert (!possible_moves.empty());
    return possible_moves.pop();
}

bool Node::is_expanded() const {
    return possible_moves.empty();
}

bool Node::is_end() const {
    return board.white == 0 || board.black == 0;
}

float get_UCT_value() const {
    if (visits == 0) return std::numeric_limits<float>::infinity();

    const float C = 1.41421356237f; // sqrt(2)
    float exploitation = static_cast<float>(score) / visits;
    float exploration = 0.0f;

    if (parent != nullptr) {
        exploration = C * std::sqrt(std::log(parent->visits) / visits);
    }

    return exploitation + exploration;
}