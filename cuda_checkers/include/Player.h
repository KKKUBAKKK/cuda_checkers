#ifndef PLAYER_H
#define PLAYER_H

#define MAX_ITERATIONS 1000
#define TIME_LIMIT_MS 1000.0f

#include "Node.h"

class Player {
private:
    Node *root;
    float time_limit_ms;
    int max_iterations;

public:
    bool is_white;
    bool is_cpu;

    explicit Player(bool is_white = true, bool is_cpu = true, int max_iterations = MAX_ITERATIONS, float time_limit_ms = TIME_LIMIT_MS) : 
    is_white(is_white), is_cpu(is_cpu), time_limit_ms(time_limit_ms), max_iterations(max_iterations) {
        root = new Node(nullptr, is_white);
    };

    Player(Board board, bool is_white = true, bool is_cpu = true, int max_iterations = MAX_ITERATIONS, float time_limit_ms = TIME_LIMIT_MS) : 
    is_white(is_white), is_cpu(is_cpu), time_limit_ms(time_limit_ms), max_iterations(max_iterations) {
        root = new Node(board, nullptr);
    };

    ~Player() {
        delete root;
    };

    int findEqualChild(Board board) {
        if (root == nullptr || root->children.empty()) {
            return -1;
        }

        for (int i = 0; i < root->children.size(); i++) {
            if (root->children[i]->board.is_equal(board)) {
                return i;
            }
        }

        return -1;
    }

    void moveRoot(Board startBoard) {
        if (root == nullptr) {
            root = new Node(startBoard, nullptr);
            return;
        }
        
        if (root->board.is_equal(startBoard)) {
            return;
        }
            
        int index = findEqualChild(startBoard);
        if (index == -1) {
            delete root;
            root = new Node(startBoard, nullptr);
            return;
        }

        Node *temp = root;
        root = root->children[index];
        temp->children.erase(temp->children.begin() + index);
        delete temp;
    }

    Node* selectNode() {
        // Select the best child
        // If the child is not fully expanded, return it
        // Otherwise, return the best child of the child
        // TODO: fix this
        Node* current = root;
        while (!current->children.empty()) {
            Node* best_child = nullptr;
            float best_value = -std::numeric_limits<float>::infinity();

            for (Node* child : current->children) {
                float uct_value = child->getUCTValue();
                if (uct_value > best_value) {
                    best_value = uct_value;
                    best_child = child;
                }
            }

            if (best_child == nullptr) {
                break;
            }

            current = best_child;
        }

        return current;
    }

    // TODO: complete this
    void mctsLoop() {
        // Run the MCTS algorithm
        for (int i = 0; i < max_iterations; i++) {
            // 1. Selection
            // Find nodes for expansion

            // 2. Expansion
            // For each node, create a new node and add it to the tree

            // 3. Simulation
            // Run simulations on the new nodes

            // 4. Backpropagation
            // Backpropagate the results up the tree
        }
    }

    // TODO: complete this
    Board makeMove(Board startBoard) {
        // Move root to the current board
        moveRoot(startBoard);

        // Run the MCTS algorithm
        mctsLoop();

        // Find the best child
        // Set the root to the best child
        // Return the child's board
    }
};

#endif // PLAYER_H