#ifndef POOLING_H
#define POOLING_H
#include "../../headers/utilitaire/Shapes.h"
#include "../../headers/layers/Layer.h"

typedef struct Pooling
{
   Layer layer;
   Shapes input_shapes;
   Shapes window_shapes;

}Pooling_s , *Pooling;


Pooling new_Reshape(Shapes input_shapes , Shapes window_shapes , 
                    void* (*forward)(void *layer ,void *inputs),
                    void* (*backward)(void *layer , void* output_gradient, double learning_rate));

void* reshape_forward(void *pooling_layer , void *inputs);
void* reshape_backward(void *pooling_layer , void *output_gradient , double learning_rate);

#endif