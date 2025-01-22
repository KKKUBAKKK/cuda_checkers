#ifndef PLAYER_H
#define PLAYER_H

#define MAX_ITERATIONS 1000
#define TIME_LIMIT_MS 1000.0f
#define MAX_GAMES 1000
#define THREADS_PER_BLOCK 256

#include "Node.h"

class Player {
private:
    Node *root;
    float time_limit_ms;
    int max_games;
    int max_iterations;

public:
    bool is_white;
    bool is_cpu;

    explicit Player(bool is_white = true, bool is_cpu = true, int max_games = MAX_GAMES, int max_iterations = MAX_ITERATIONS, float time_limit_ms = TIME_LIMIT_MS) :
    is_white(is_white), is_cpu(is_cpu), time_limit_ms(time_limit_ms), max_iterations(max_iterations), max_games(max_games) {
        root = new Node(nullptr, is_white);
    };

    Player(Board board, bool is_white = true, bool is_cpu = true, int max_games = MAX_GAMES, int max_iterations = MAX_ITERATIONS, float time_limit_ms = TIME_LIMIT_MS) :
    is_white(is_white), is_cpu(is_cpu), time_limit_ms(time_limit_ms), max_iterations(max_iterations), max_games(max_games) {
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

    Node* select() {
        // Select the best child
        // If the child is not fully expanded, return it
        // Otherwise, return the best child of the child
        Node* current = root;
        while (current->is_expanded() && !current->is_end()) {
            Node* best_child = nullptr;
            float best_value = -std::numeric_limits<float>::infinity();

            for (Node* child : current->children) {
                float uct_value = child->get_UCT_value();
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

    Node* expand(Node* node) {
        // Expand the node
        // Create a new node and add it to the tree
        // Return the new node
        Move new_move = node->get_move();
        Board new_board = node->board.apply_move(new_mvoe);
        Node* new_node = new Node(new_board, node);
        node->children.push_back(new_node);
        return new_node;
    }

    int simulate(Node *node) {
        // Simulate the game
        // Run the game until the end
        // Return the result
        if (is_cpu)
            return simulate_cpu(node->board);

        return simulate_gpu(node->board);
    }

    int simulate_cpu(Board board) {
        // Simulate n games on the CPU
        // Run the game until the end
        // Return the result
        Move *moves = new Move[MAX_MOVES];
        Move *stack = new Move[MAX_MOVES];
        std::random_device rd;
        std::mt19937 rng(rd());
        int result = root->board.simulate_n_games_cpu(rng, moves, stack, max_games, time_limit_ms);
        delete[] moves;
        delete[] stack;
        return result;
    }

    int simulate_gpu(Board board) {
        // Simulate n games on the GPU
        // Run the game until the end
        // Return the result
        int num_blocks = (max_games + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        // Allocate memory for results on the GPU
        // int* d_results;
        // cudaMalloc(&d_results, max_games * sizeof(int));
        // cudaMemset(d_results, 0, max_games * sizeof(int));
        int* d_results;
        cudaMalloc(&d_results, sizeof(int));
        cudaMemset(d_results, 0, sizeof(int));

        // Allocate memory for random states on the GPU
        curandState* d_states;
        cudaMalloc(&d_states, max_games * sizeof(curandState));

        // Initialize random states
        init_curand<<<num_blocks, THREADS_PER_BLOCK>>>(d_states, time(NULL));

        // Launch the kernel
        simulate_game_gpu_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(board, d_results, d_states);

        // Copy results back to the host
        // int* h_results = new int[max_games];
        // cudaMemcpy(h_results, d_results, max_games * sizeof(int), cudaMemcpyDeviceToHost);
        int h_results;
        cudaMemcpy(&h_results, d_results, sizeof(int), cudaMemcpyDeviceToHost);

        // Sum up the results
        // int total_result = 0;
        // for (int i = 0; i < max_games; i++) {
        //     total_result += h_results[i];
        // }

        // Free allocated memory
        // delete[] h_results;
        cudaFree(d_results);
        cudaFree(d_states);

        return total_result;
    }

    __global__ void simulate_game_gpu_kernel(Board initial_board, int* results, curandState* states) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        Move moves[MAX_MOVES];
        Move stack[MAX_MOVES];

        // Each thread gets its own random state
        curandState localState = states[tid];

        // Run simulation
        int result = initial_board.simulate_game_gpu(&localState, moves, stack);

        // Store result
        atomicAdd(&results, result);

        // Save updated random state
        states[tid] = localState;
    }

    void backpropagate(Node *node, int score) {
        // Backpropagate the results up the tree
        // Update the score and visits of each node
        Node *current = node;
        while (current != nullptr) {
            current->score += score;
            current->visits++;
            current = current->parent;
        }
    }

    int mctsLoop() {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto time_limit = std::chrono::milliseconds(time_limit_ms);

        // Run the MCTS algorithm
        int i = 0;
        for (i = 0; i < max_iterations; i++) {
            auto current_time = std::chrono::high_resolution_clock::now();
            if (current_time - start_time >= time_limit) {
                break;
            }

            // 1. Selection
            // Find nodes for expansion
            Node *selected = select();

            // 2. Expansion
            // For each node, create a new node and add it to the tree
            Node *new_node = expand(selected);

            // 3. Simulation
            // Run simulations on the new nodes
            int score = simulate(new_node);

            // 4. Backpropagation
            // Backpropagate the results up the tree
            backpropagate(new_node, score);
        }

        return i;
    }

    // TODO: fix calculating score in simulations
    // TODO: check if it should be always positive
    // TODO: it should probably depend on the color of the player, not the board
    Node *choose_move() {
        // Find the best child
        Node* best_child = nullptr;
        int best_score = -1;

        for (Node* child : root->children) {
            if (child->score > best_score) {
                best_score = child->score;
                best_child = child;
            }
        }

        return best_child;
    }

    Board makeMove(Board startBoard) {
        // Move root to the current board
        moveRoot(startBoard);

        // Run the MCTS algorithm
        mctsLoop();

        // Find the best child
        Node* best_child = choose_move();

        // Set the root to the best child
        moveRoot(best_child->board);

        // Return the child's board
        return best_child->board;
    }
};

#endif // PLAYER_H