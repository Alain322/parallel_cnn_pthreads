#ifndef DENSE_H
#define DENSE_H

#include "../arrays/Arrayc.h"
#include "Layer.h"

typedef struct Dense
{
    Layer layer;//herite de la structure de base layer

    Array weights;
    Array bias;
} Dense_s , *Dense;

Dense new_dense(int input_size , int output_size , float init_min_weights , float init_max_weights
                ,void* (*forward)(void *layer ,void *inputs),
                void* (*backward)(void *layer , void* output_gradient, double learning_rate)
                );

void* dense_forward(void *dense_layer , void *inputs);
void* dense_backward(void *dense_layer , void *output_gradient , double learning_rate);
void free_dense_layer(Dense dense_layer);

#endif