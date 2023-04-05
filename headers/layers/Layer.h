#ifndef LAYER_H
#define LAYER_H
#include<pthread.h>




typedef struct Layer
{
    void *inputs;
    void *output;

    void *child_layer;
    pthread_mutex_t mutex_layer;

    void* (*forward)(void *layer , void *inputs);
    void* (*backward)(void *layer , void* output_gradient, double learning_rate);
}Layer_s , *Layer;


Layer new_Layer(void* (*forward)(void *layer ,void *inputs),
                void* (*backward)(void *layer , void* output_gradient, double learning_rate));

#endif