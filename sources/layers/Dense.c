#include<stdlib.h>

#include "../../headers/layers/Dense.h"
#include "../../headers/layers/Layer.h"
#include "../../headers/arrays/Arrayc.h"

Dense new_dense(int input_size , int output_size , float init_min_weights , float init_max_weights 
                ,void* (*forward)(void *layer ,void *inputs),
                void* (*backward)(void *layer , void* output_gradient, double learning_rate)
                ){
    //contruction de la structure layer 
    //super                
    Layer layer = new_Layer(forward , backward);
    
    Dense dense = (Dense) calloc(1 , sizeof(Dense_s));
    dense->layer = layer;
    dense->layer->child_layer = dense;
    dense->weights = randomArray(output_size , input_size , init_min_weights , init_max_weights);
    dense->bias = randomArray(output_size , 1 , init_min_weights , init_max_weights);
    return dense;
}

void* dense_forward(void *dense_layer , void *inputs){
    Dense dense = (Dense) dense_layer;
    dense->layer->inputs = inputs;
    return matMulArray(dense->weights , (Array) inputs);
}



void* dense_backward(void *dense_layer , void *output_gradient , double learning_rate){
    // printf("enter back dense\n");
    //je dois libere l'ancienne entrer
    Dense dense = (Dense) dense_layer;
    Array input_T = transposeArray(dense->layer->inputs);
    pthread_mutex_lock(&dense->layer->mutex_layer);
    // printfArray((Array) dense->layer->inputs , True);
    // printfArray(dense->weights , True);
    Array weights_gradient = matMulArray((Array) output_gradient, input_T);

    //equivalent to weights_gradient = weights_gradient * learning_rate
    constOpApplyArray_r(weights_gradient , 
                        weights_gradient , //weights_gradient * learning_rate
                        -learning_rate , mul_Array);

    Array weight_T = transposeArray(dense->weights);

    Array input_gradient = matMulArray(weight_T , output_gradient);


    // dense->weights 
    plusArray_r(dense->weights ,  
                weights_gradient,
                dense->weights //weights + weights_gradient
                );
    pthread_mutex_unlock(&dense->layer->mutex_layer);
    freeArray(input_T);
    freeArray(weight_T);
    freeArray(weights_gradient);
    freeArray(output_gradient);
    return input_gradient;
}

void free_dense_layer(Dense dense_layer){
    freeArray(dense_layer->weights);
    freeArray(dense_layer->bias);
    free(dense_layer->layer);
    free(dense_layer);
}