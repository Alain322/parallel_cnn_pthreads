#include "../../headers/layers/Activation.h"
#include "../../headers/arrays/Arrayc.h"

#include <stdlib.h>

Activation new_Activation(int inputs_numbers ,void* (*activation_function)(void *inputs),
                    void* (*activation_prime)(void *input),
                    void* (*forward)(void *layer ,void *inputs),
                    void* (*backward)(void *layer , void* output_gradient, double learning_rate)){

    //on fait appele au contruction de la structure layer 
    //super                
    Layer layer = new_Layer(forward , backward);
    Activation activation = (Activation) calloc(1 , sizeof(Activation_s));
    activation->layer = layer;
    activation->layer->child_layer = activation;
    activation->activation_function = activation_function;
    activation->activation_prime = activation_prime;
    activation->inputs_numbers = inputs_numbers;

    return activation;
}

void* activation_forward(void *activation_layer , void *inputs){
    Activation act_layer = (Activation) activation_layer;
    pthread_mutex_lock(&act_layer->layer->mutex_layer);
    act_layer->layer->inputs = inputs;
    pthread_mutex_unlock(&act_layer->layer->mutex_layer);
    return act_layer->activation_function(act_layer->layer->inputs);
}


void* activation_backward(void *activation_layer , void *output_gradient , double learning_rate){
    // printf("enter sdhjgjhskdfsd\n");
    Activation act_layer = (Activation) activation_layer;
    pthread_mutex_lock(&act_layer->layer->mutex_layer);
    // printfArray((Array) act_layer->layer->inputs , True);
    // printf("enter back activation\n");
    Array prime = (Array) act_layer->activation_prime(act_layer->layer->inputs);
    Array result = mulArray(output_gradient , prime);
    freeArray(prime);
    freeArray(output_gradient);
    pthread_mutex_unlock(&act_layer->layer->mutex_layer);
    return result;
}

void* activation_forward_depth(void *activation_layer , void *inputs){
    Activation act_layer = (Activation) activation_layer;
    pthread_mutex_lock(&act_layer->layer->mutex_layer);
    int i = 0;
    // printf("enter %d\n" , act_layer->inputs_numbers);
    Array *result = calloc(act_layer->inputs_numbers , sizeof(Array));
    for (i = 0; i < act_layer->inputs_numbers; i++)
    {
        // printfArray(((Array*) inputs)[i] , True);
        result[i] = act_layer->activation_function(((Array*) inputs)[i]);
    }
    
    act_layer->layer->inputs = inputs;
    pthread_mutex_unlock(&act_layer->layer->mutex_layer);
    return result;
}


void* activation_backward_depth(void *activation_layer , void *output_gradient , double learning_rate){
    // printf("enter sdhjgjhskdfsd\n");
    Activation act_layer = (Activation) activation_layer;
    pthread_mutex_lock(&act_layer->layer->mutex_layer);
    // printfArray((Array) act_layer->layer->inputs , True);
    // printf("enter back activation\n");
    int i = 0;
    Array *result_tab = calloc(act_layer->inputs_numbers , sizeof(Array));
    for (i = 0; i < act_layer->inputs_numbers; i++)
    {
        Array prime = (Array) act_layer->activation_prime(((Array*) act_layer->layer->inputs)[i]);
        result_tab[i] = mulArray(((Array*) output_gradient)[i] , prime);
        freeArray(prime);
    }
    
    // Array result = mulArray(output_gradient , prime);
    // freeArray(prime);
    for (i = 0; i < act_layer->inputs_numbers; i++)
    {
        freeArray(((Array*) output_gradient)[i]);
    }
    pthread_mutex_unlock(&act_layer->layer->mutex_layer);
    freeArray(output_gradient);
    return result_tab;
}








