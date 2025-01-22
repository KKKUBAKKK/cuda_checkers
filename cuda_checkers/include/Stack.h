#include <cstdint>
#include "Board.h"

#ifndef STACK_H
#define STACK_H

// #define STACK_SIZE 64

class Stack {
public:
    Move *stack;
    int top; // Top is first empty position
    int n;

    __host__ __device__ explicit Stack(Move *stack = nullptr, int n = MAX_MOVES) {
        top = 0;
        this->stack = stack;
        this->n = n;
        if (this->stack == nullptr) {
            Move temp[MAX_MOVES];
            this->stack = temp;
            this->n = MAX_MOVES;
        }
    };

    __host__ __device__ ~Stack() {
        // delete[] stack;
    };

    __host__ __device__ void push(Move value) {
        assert(top < n);
        stack[top++] = value;
    };

    __host__ __device__ Move pop() {
        assert(top > 0);
        return stack[--top];
    };

    __host__ __device__ bool is_empty() {
        return top == 0;
    };

    __host__ __device__ bool is_full() {
        return top == n;
    };

    __host__ __device__ Move peek() {
        assert(top > 0);
        return stack[top - 1];
    };

    __host__ __device__ int size() {
        return top;
    };
};

#endif // STACK_H