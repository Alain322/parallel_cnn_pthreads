#ifndef ACTIVATION_H
#define ACTIVATION_H
#include "Layer.h"

typedef struct Activation
{
    Layer layer; //herite de la structure de base layer
    int inputs_numbers;
    void* (*activation_function)(void *inputs);
    void* (*activation_prime)(void *input);
}Activation_s , *Activation;


Activation new_Activation(int input_numbers , void* (*activation_function)(void *inputs),
                    void* (*activation_prime)(void *input),
                    void* (*forward)(void *layer ,void *inputs),
                    void* (*backward)(void *layer , void* output_gradient, double learning_rate) 
            );


void* activation_forward(void *activation_layer , void *inputs);
void* activation_backward(void *activation_layer , void *output_gradient , double learning_rate);

void* activation_forward_depth(void *activation_layer , void *inputs);
void* activation_backward_depth(void *activation_layer , void *output_gradient , double learning_rate);

#endif