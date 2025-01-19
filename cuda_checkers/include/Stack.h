#include <cstdint>
#include "Move.h"

#ifndef STACK_H
#define STACK_H

#define STACK_SIZE 64

class Stack {
public:
    Move stack[STACK_SIZE];
    int top; // Top is first empty position

    __host__ __device__ explicit Stack() {
        top = 0;
    };

    __host__ __device__ ~Stack() {
        // delete[] stack;
    };

    __host__ __device__ void push(Move value) {
        stack[top++] = value;
    };

    __host__ __device__ Move pop() {
        return stack[--top];
    };

    __host__ __device__ bool is_empty() {
        return top == 0;
    };

    __host__ __device__ bool is_full() {
        return top == STACK_SIZE;
    };

    __host__ __device__ Move peek() {
        return stack[top - 1];
    };

    __host__ __device__ int size() {
        return top;
    };
};

#endif // STACK_H