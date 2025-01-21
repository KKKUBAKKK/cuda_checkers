#ifndef PLAYER_H
#define PLAYER_H

#include "Node.h"

class Player {
private:
    Node *root;

public:
    bool is_white;
    bool is_cpu;

    explicit Player(bool is_white = true, bool is_cpu = true) : is_white(is_white), is_cpu(is_cpu) {
        root = new Node(nullptr, is_white);
    };

    Player(Board board, bool is_white = true, bool is_cpu = true) : is_white(is_white), is_cpu(is_cpu) {
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

    Board makeMove(Board startBoard) {
        moveRoot(startBoard);
    }
};

#endif // PLAYER_H