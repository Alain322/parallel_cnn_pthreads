#include "../../headers/layers/softmax.h"
#include "../../headers/arrays/Arrayc.h"
#include "../../headers/utilitaire/utilitaire.h"
#include <math.h>

#include <stdlib.h>

Softmax new_Softmax(void* (*forward)(void *layer ,void *inputs),
                       void* (*backward)(void *layer , void* output_gradient, double learning_rate)){
    
    //contruction de la structure layer 
    //super                
    Layer layer = new_Layer(forward , backward);
    Softmax softmax = calloc(1 , sizeof(Softmax_s));
    softmax->layer = layer;
    softmax->layer->child_layer = softmax;
    return softmax;
}


void* softmax_forward(void *softmax_layer , void *inputs){
    Softmax soft_layer = (Softmax) softmax_layer;
    pthread_mutex_lock(&soft_layer->layer->mutex_layer);
    Array temp = op_Array(inputs , pow2);
    constOpApplyArray_r(temp , temp , sumElementArray(temp) , div_Array);
    soft_layer->layer->output = temp;
    // printfArray(soft_layer->layer->output , True);
    pthread_mutex_unlock(&soft_layer->layer->mutex_layer);
    return soft_layer->layer->output;
}


void* softmax_backward(void *softmax_layer , void *output_gradient , double learning_rate){
    // printf("enter back softmax\n");
    // Array result = NULL;
    Softmax soft_layer = (Softmax) softmax_layer;
    // printfArray((Array) soft_layer->layer->output , True);
    // printf("------------------------------------------\n");

    int size = ((Array) soft_layer->layer->output)->nRow * ((Array) soft_layer->layer->output)->nCol;
    Array identity = identityArray(size);
    Array output_T = transposeArray((Array) soft_layer->layer->output);
    Array temp1 = vectOpApplyArray(identity , output_T , minus_Array); // identity - output_T
    Array temp2 = mulArray(temp1 , (Array) soft_layer->layer->output);// (identity - output_T)*output
    Array result = matMulArray(temp2 , (Array) output_gradient); //dot((identity - output_T)*output , output_gradient)

    freeArray(temp1);
    freeArray(temp2);
    freeArray(output_T);
    freeArray(identity);
    // freeArray(output_gradient);

    return result;
}