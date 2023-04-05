#include "../../headers/layers/Layer.h"
#include<stdlib.h>
#include<pthread.h>


Layer new_Layer(void* (*forward)(void *layer ,void *inputs),
                void* (*backward)(void *layer , void* output_gradient, double learning_rate)){

    
    Layer layer = (Layer) calloc(1 , sizeof(Layer_s));
    layer->forward = forward;
    layer->backward = backward;
    layer->child_layer = NULL;
    layer->inputs = NULL;
    layer->output = NULL;
    pthread_mutex_init(&layer->mutex_layer, NULL);
    return layer;
}

