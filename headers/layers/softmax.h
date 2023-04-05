#ifndef SOFTMAX_H
#define SOFTMAX_H
#include "Layer.h"

typedef struct Softmax
{
    Layer layer; //herite de la structure de base layer
}Softmax_s , *Softmax;


Softmax new_Softmax(void* (*forward)(void *layer ,void *inputs),
                       void* (*backward)(void *layer , void* output_gradient, double learning_rate) 
                    );


void* softmax_forward(void *softmax_layer , void *inputs);
void* softmax_backward(void *softmax_layer , void *output_gradient , double learning_rate);



#endif