#include "Player.h"
#include <chrono>
#include <iostream>

__global__ void simulate_game_gpu_kernel(int *is_white_move, uint32_t *white, uint32_t *black, uint32_t *queens, float* results, curandState* states, bool *is_player_white, int *max_games) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < *max_games) {
        Move moves[MAX_MOVES];
        Move stack[MAX_MOVES];

        // Each thread gets its own random state
        curandState localState = states[tid];

        Board initial_board(is_white_move);
        initial_board.white = *white;
        initial_board.black = *black;
        initial_board.queens = *queens;

        // Run simulation
        float result = initial_board.simulate_game_gpu(&localState, moves, stack, *is_player_white);

        // Store result
        atomicAdd(results, result);

        // Save updated random state
        states[tid] = localState;
    }
}

__global__ void init_curand(curandState* state, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

Player::Player(bool is_white, bool is_cpu, int max_games, float time_limit_ms) :
                        is_white(is_white), is_cpu(is_cpu), time_limit_ms(time_limit_ms), max_games(max_games) {
    root = new Node(nullptr, true);

    // Initialize random states
    if (!is_cpu) {
        CUDA_CHECK(cudaMalloc(&states, max_games * sizeof(curandState)));
        int num_blocks = (max_games + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        init_curand<<<num_blocks, THREADS_PER_BLOCK>>>(states, time(NULL));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        states = nullptr;
    }
}

Player::Player(Board board, bool is_white, bool is_cpu, 
    int max_games, float time_limit_ms) :
    is_white(is_white), is_cpu(is_cpu), time_limit_ms(time_limit_ms), max_games(max_games) {
    root = new Node(board, nullptr);

    // Initialize random states
    if (!is_cpu) {
        CUDA_CHECK(cudaMalloc(&states, max_games * sizeof(curandState)));
        int num_blocks = (max_games + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        init_curand<<<num_blocks, THREADS_PER_BLOCK>>>(states, time(NULL));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        states = nullptr;
    }
};

Player::~Player() {
    delete root;
    if (states != nullptr)
        CUDA_CHECK(cudaFree(states));
};

int Player::findEqualChild(Board board) {
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

void Player::move_root(Board startBoard) {

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
    root->parent = nullptr;
    temp->children[index] = nullptr;
    delete temp;
}

Node* Player::select() {
    // Select the best child
    // If the child is not fully expanded, return it
    // Otherwise, return the best child of the child
    Node* current = root;
    while (current->is_expanded() && !current->is_end()) {
        Node* best_child = nullptr;
        float best_value = -std::numeric_limits<float>::infinity();

        for (Node* child : current->children) {
            if (child == nullptr) continue;
            float uct_value = child->get_UCT_value();
            if (uct_value > best_value) {
                best_value = uct_value;
                best_child = child;
            }
        }
         
        assert(best_child != nullptr);

        current = best_child;
    }

    return current;
}

Node* Player::expand(Node* node) {
    // Expand the node
    // Create a new node and add it to the tree
    // Return the new node
    Move new_move = node->get_move();
    Board new_board = node->board.apply_move(new_move);
    Node* new_node = new Node(new_board, node);
    node->children.push_back(new_node);
    return new_node;
}

float Player::simulate(Node *node) {
    // Simulate the game
    // Run the game until the end
    // Return the result
    if (is_cpu) {
        return simulate_cpu(node->board);
    }

    return simulate_gpu(node->board);
}

float Player::simulate_cpu(Board board) {
    // Simulate n games on the CPU
    // Run the game until the end
    // Return the result
    Move *moves = new Move[MAX_MOVES];
    Move *stack = new Move[MAX_MOVES];
    std::random_device rd;
    std::mt19937 rng(rd());
    float result = root->board.simulate_n_games_cpu(rng, moves, stack, is_white, max_games, time_limit_ms);
    delete[] moves;
    delete[] stack;
    return result;
}

float Player::simulate_gpu(Board board) {
    // Simulate n games on the GPU
    // Run the game until the end
    // Return the result
    int num_blocks = (max_games + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Allocate memory for the board
    int *is_white_move;
    uint32_t *white;
    uint32_t *black;
    uint32_t *queens;
    CUDA_CHECK(cudaMalloc(&is_white_move, sizeof(int)));
    CUDA_CHECK(cudaMemset(is_white_move, board.whiteToMove, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&white, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(white, board.white, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&black, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(black, board.black, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&queens, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(queens, board.queens, sizeof(uint32_t)));

    // Allocate memory for results on the GPU
    float* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_results, 0.0f, sizeof(float)));

    // Allocate memory for max_games on the GPU
    int *d_max_games;
    CUDA_CHECK(cudaMalloc(&d_max_games, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_max_games, max_games, sizeof(int)));

    // Allocate memory for is_white
    bool *d_is_white;
    CUDA_CHECK(cudaMalloc(&d_is_white, sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_is_white, is_white, sizeof(bool)));

    // Launch the kernel
    simulate_game_gpu_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(is_white_move, white, black, queens, d_results, states, d_is_white, d_max_games);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to the host
    float h_results;
    CUDA_CHECK(cudaMemcpy(&h_results, d_results, sizeof(float), cudaMemcpyDeviceToHost));

    // Free allocated memory
    CUDA_CHECK(cudaFree(d_max_games));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_is_white));
    CUDA_CHECK(cudaFree(is_white_move));
    CUDA_CHECK(cudaFree(white));
    CUDA_CHECK(cudaFree(black));
    CUDA_CHECK(cudaFree(queens));


    return h_results;
}

void Player::backpropagate(Node *node, float score) {
    // Backpropagate the results up the tree
    // Update the score and visits of each node
    Node *current = node;
    while (current != nullptr) {
        current->score += score;
        current->visits += max_games;
        current = current->parent;
    }
}

int Player::mcts_loop() {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto time_limit = std::chrono::milliseconds((int) time_limit_ms);

    // Run the MCTS algorithm
    int i = 0;
    while (true) {
        auto current_time = std::chrono::high_resolution_clock::now();
        if (current_time - start_time >= time_limit) {
            break;
        }

        // 1. Selection
        // Find nodes for expansion
        Node *selected = select();
        if (selected->is_end()) {
            float score = selected->white_score();
            assert (score >= 0);
            assert (score <= 1);
            if (!is_white) score = 1 - score;
            backpropagate(selected, score * max_games);
            continue;
        }

        // 2. Expansion
        // For each node, create a new node and add it to the tree
        Node *new_node = expand(selected);

        // 3. Simulation
        // Run simulations on the new nodes
        float score = simulate(new_node);
        assert (score >= 0);
        assert (score <= max_games);

        // 4. Backpropagation
        // Backpropagate the results up the tree
        backpropagate(new_node, score);

        i++;
    }
    std::cerr << "Loop it iterations: " << i << std::endl;
    return i;
}

Node* Player::choose_move() {
    // Find the best child
    Node* best_child = nullptr;
    int best_score = -1;

    for (Node* child : root->children) {
        if (child->score > best_score) {
            best_score = child->score / child->visits;
            best_child = child;
        }
    }

    return best_child;
}

Board Player::make_move(Board start_board) {
    // Move root to the current board
    move_root(start_board);

    // If root doesn't have any children, return the current board
    if (root->is_end()) {
        return start_board;
    }

    // Run the MCTS algorithm
    mcts_loop();

    // Find the best child
    Node* best_child = choose_move();
    assert(best_child != nullptr);

    // Set the root to the best child
    move_root(best_child->board);

    // Return the child's board
    return best_child->board;
}