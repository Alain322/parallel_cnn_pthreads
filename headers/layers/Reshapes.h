#ifndef RESHAPES_H
#define RESHAPES_H
#include "../../headers/utilitaire/Shapes.h"
#include "../../headers/layers/Layer.h"

typedef struct Reshape
{
   Layer layer;
   Shapes input_shapes;
//    Shapes output_shapes;

}Reshape_s , *Reshape;


Reshape new_Reshape(Shapes input_shapes ,
                    void* (*forward)(void *layer ,void *inputs),
                    void* (*backward)(void *layer , void* output_gradient, double learning_rate));

void* reshape_forward(void *reshape_layer , void *inputs);
void* reshape_backward(void *reshape_layer , void *output_gradient , double learning_rate);

#endif