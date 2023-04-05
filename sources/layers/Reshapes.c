#include <stdlib.h>
#include "../../headers/layers/Reshapes.h"
#include "../../headers/arrays/Arrayc.h"


Reshape new_Reshape(Shapes input_shapes ,  
                    void* (*forward)(void *layer ,void *inputs),
                    void* (*backward)(void *layer , void* output_gradient, double learning_rate)){
    
    //contruction de la structure layer 
    //super                
    Layer layer = new_Layer(forward , backward);
    Reshape reshape = calloc(1 , sizeof(Reshape_s));
    reshape->layer = layer;
    reshape->layer->child_layer = reshape;


    reshape->layer->forward = forward;
    reshape->layer->backward = backward;

    reshape->input_shapes = input_shapes;
    // reshape->output_shapes = output_shapes;

    return reshape;
}

void* reshape_forward(void *reshape_layer , void *inputs){

    Reshape reshape_l = (Reshape) reshape_layer;
    Array *input_tab = (Array*) inputs;
    pthread_mutex_lock(&reshape_l->layer->mutex_layer);

    Array output_array = calloc(1 , sizeof(Reshape_s));
    int kernel_size = reshape_l->input_shapes.height * reshape_l->input_shapes.width;
    int output_len = reshape_l->input_shapes.depth * kernel_size;
    output_array->nRow =  output_len;
    output_array->nCol = 1;
    output_array->data = create_matrix_double(output_len  , 1);

    
    // __parameter_flatten_cnn pa = (__parameter_flatten_cnn)layer->parameters;
    // int p1 = pa->height_input * pa->width_input;
    // int output_len = pa->nbre_card_input * p1;
    // printf("enter\n");
    int kernel_width = reshape_l->input_shapes.width;
    int i, j, k;
    // output = create_vector(output_len);
    for (i = 0; i < reshape_l->input_shapes.depth; i++)
        for (j = 0; j < reshape_l->input_shapes.height; j++)
            for (k = 0; k < reshape_l->input_shapes.width; k++)
                output_array->data[i * kernel_size + j * kernel_width + k][0] = input_tab[i]->data[j][k];
    
    // printfArray(output_array , True);
    pthread_mutex_unlock(&reshape_l->layer->mutex_layer);			
    return output_array;
}


void* reshape_backward(void *reshape_layer , void *output_gradient , double learning_rate){
    Reshape reshape_l = (Reshape) reshape_layer;

    pthread_mutex_lock(&reshape_l->layer->mutex_layer);

    Array output_grad = (Array) output_gradient;

    // printfArray(output_grad , True);

    // double ***error_input = NULL;
    Array *error = calloc(reshape_l->input_shapes.depth , sizeof(Array));
    // __parameter_flatten_cnn pa = (__parameter_flatten_cnn)layer->parameters;

    // int p1 = pa->height_input * pa->width_input;
    int kernel_size = reshape_l->input_shapes.height * reshape_l->input_shapes.width; 

    int output_len = kernel_size * reshape_l->input_shapes.depth;

    int i, j, k, s, s_;
    // error_input = malloc(pa->nbre_card_input * sizeof(double **));
   

    // for (i = 0; i < pa->nbre_card_input; i++)
    //     error_input[i] = create_matrix(pa->height_input, pa->width_input);

    for (i = 0; i < reshape_l->input_shapes.depth; i++)
    {
        error[i] = zerosArray(reshape_l->input_shapes.height , reshape_l->input_shapes.width);
    }
    

    for (s = 0; s < output_len; s++)
    {
        s_ = s;
        i = s_ / kernel_size;
        s_ = s_ % kernel_size;
        j = s_ / reshape_l->input_shapes.width;
        s_ = s_ % reshape_l->input_shapes.width;
        k = s_;

        error[i]->data[j][k] = output_grad->data[s][0];
    }
    
    pthread_mutex_unlock(&reshape_l->layer->mutex_layer);
    freeArray((Array) output_gradient);
    
    return error;
}