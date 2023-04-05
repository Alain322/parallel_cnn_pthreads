#ifndef CONVOLUTION_H
#define CONVOLUTION_H
#include "Layer.h"
#include "../../headers/arrays/Arrayc.h"
#include "../../headers/utilitaire/Shapes.h"

typedef struct Convolution_Layer
{
    Layer layer;

    Shapes input_shapes;
    Shapes output_shapes;
    Shapes kernel_shapes;
    Array **kernels;
    Array *biases;
    int depth;

}Convolution_Layer_s , *Convolution_Layer;

Convolution_Layer new_Convolution_Layer(Shapes input_shapes , Shapes kernel_layer , int depth 
                ,void* (*forward)(void *layer ,void *inputs),
                void* (*backward)(void *layer , void* output_gradient, double learning_rate));



void* convolution_forward(void *layer , void *inputs);
void* convolution_backward(void *layer , void *output_gradient , double learning_rate);

Array rotate_kernel_180(Array kernel);
Array cross_corolation(Array input , Array kernel);
// Array valid_convolution(Array input , Array kernel);
Array full_convolotion_180(Array  input , Array kernel);

void free_convolution_layer(Convolution_Layer convolition_layer);


#endif