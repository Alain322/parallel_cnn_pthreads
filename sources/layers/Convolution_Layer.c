#include <stdlib.h>
#include "../../headers/utilitaire/Shapes.h"
#include "../../headers/layers/Convolution_Layer.h"
#include "../../headers/arrays/base.h"
#include "../../headers/arrays/Arrayc.h"


Convolution_Layer new_Convolution_Layer(Shapes input_shapes , Shapes kernel_shapes , int depth , 
                                        void* (*forward)(void *layer ,void *inputs),
                                        void* (*backward)(void *layer , void* output_gradient, double learning_rate)){
    
    //contruction de la structure layer 
    //super                
    Layer layer = new_Layer(forward , backward);
    Convolution_Layer convolution = calloc(1 , sizeof(Convolution_Layer_s));
    convolution->layer = layer;
    convolution->layer->child_layer = convolution;
    convolution->layer->forward = forward;
    convolution->layer->backward = backward;

    convolution->input_shapes = input_shapes;
    convolution->kernel_shapes = kernel_shapes;
    convolution->kernel_shapes.depth = input_shapes.depth;
    convolution->depth = depth;
    int output_height = input_shapes.height - kernel_shapes.height + 1;
    int output_width = input_shapes.width - kernel_shapes.width + 1;
    convolution->output_shapes = (Shapes){depth , output_height, output_width};

    int i = 0 , j = 0;
    convolution->biases = (Array*) calloc(depth , sizeof(Array));

    for (i = 0; i < depth; i++)
    {
        convolution->biases[i] = randomArray(output_height , output_width , 0 , 1);
        // convolution->biases[i] = zerosArray(output_height , output_width);
        int k = 0 , q = 0;
        for(k = 0; k < convolution->output_shapes.height; k++){
            for (q = 0; q < convolution->output_shapes.width; q++)
            {
                convolution->biases[i]->data[k][q] += 1; 
                // printf("%f" , conv_layer->biases[i]->data[k][q]);
            }
                // printfArray(conv_layer->biases[i] , )
        }
    }


    convolution->kernels = (Array**) calloc(input_shapes.depth , sizeof(Array*));
    for (j = 0; j < input_shapes.depth; j++)
    {
        convolution->kernels[j] = (Array*) calloc(depth , sizeof(Array));
    }
    
    for (i = 0; i < input_shapes.depth; i++)
    {
        for (j = 0; j < depth; j++)
        {
            convolution->kernels[i][j] = randomArray(kernel_shapes.height , kernel_shapes.width , 0 , 1);
        }
    }
    
    return convolution;
}

void* convolution_forward(void *layer , void *inputs){
    Convolution_Layer conv_layer = (Convolution_Layer) layer;
    Array *input_tabs = (Array*) inputs;

    pthread_mutex_lock(&conv_layer->layer->mutex_layer);

    conv_layer->layer->inputs = input_tabs;
    Array *output_tab = calloc(conv_layer->depth , sizeof(Array));
    int i = 0 , j = 0;

    for (j = 0; j < conv_layer->depth; j++)
    {
        Array array_result = zerosArray(conv_layer->output_shapes.height , conv_layer->output_shapes.width);
        // array_result->data = create_matrix_double(conv_layer->output_shapes.height , conv_layer->output_shapes.width);
        // array_result->nRow = conv_layer->output_shapes.height;
        // array_result->nCol = conv_layer->output_shapes.width;
        for (i = 0; i < conv_layer->input_shapes.depth; i++)
        {   
            Array temp = cross_corolation(input_tabs[i] , conv_layer->kernels[i][j]);
            plusArray_r(array_result , temp , array_result);
            freeArray(temp);
        }
        // plusArray_r(array_result , conv_layer->biases[j] , array_result);
        output_tab[j] = array_result;
    }

    conv_layer->layer->output = output_tab;

    pthread_mutex_unlock(&conv_layer->layer->mutex_layer);
    return conv_layer->layer->output;
}


void* convolution_backward(void *layer , void *output_gradient , double learning_rate){
    int i = 0 , j = 0;
    // printf("-------------------biases-------------------------\n");
    Convolution_Layer conv_layer = (Convolution_Layer) layer;
    pthread_mutex_lock(&conv_layer->layer->mutex_layer);
    Array **kernels_gradient = (Array**) calloc(conv_layer->input_shapes.depth , sizeof(Array*));

    for (i = 0; i < conv_layer->input_shapes.depth; i++)
    {
        kernels_gradient[i] = (Array*) calloc(conv_layer->depth , sizeof(Array));
    }

    Array *inputs_gradients = calloc(conv_layer->input_shapes.depth , sizeof(Array));

    for (j = 0; j < conv_layer->depth; j++)
    {   
        Array array_result = zerosArray(((Array*) output_gradient)[j]->nRow + conv_layer->kernel_shapes.height - 1
                                        ,((Array*) output_gradient)[j]->nCol + conv_layer->kernel_shapes.width - 1);
        for (i = 0; i < conv_layer->input_shapes.depth; i++)
        {
            kernels_gradient[i][j] = cross_corolation(((Array*) conv_layer->layer->inputs)[i] , ((Array*) output_gradient)[j]);
            constOpApplyArray_r(kernels_gradient[i][j] , kernels_gradient[i][j] , -learning_rate , mul_Array);
            Array temp = full_convolotion_180(((Array*) output_gradient)[j] , conv_layer->kernels[i][j]);
            plusArray_r(array_result , temp , array_result);
            freeArray(temp);
        }
        inputs_gradients[j] = array_result;
    }

    //mise a jour des poids des noyaux
    for (i = 0; i < conv_layer->input_shapes.depth; i++)
    {
        for (j = 0; j < conv_layer->depth; j++)
        {
            plusArray_r(conv_layer->kernels[i][j] , kernels_gradient[i][j] , conv_layer->kernels[i][j]);
            freeArray(kernels_gradient[i][j]);
        }
    }

    //mise a jour des biais
    int k , q;
    // printf("---------------------------ljsidjpdsfo---------------------%d" , conv_layer->output_shapes.depth);
    for (i = 0; i < conv_layer->output_shapes.depth; i++)
    {
        // printfArray(((Array*) output_gradient)[i] , True);
        Array temp = constOpApplyArray(((Array*) output_gradient)[i] , -learning_rate , mul_Array);
        plusArray_r(conv_layer->biases[i] , temp , conv_layer->biases[i]);
        freeArray(temp);
    }
    

    for (i = 0; i < conv_layer->depth; i++)
    {
        freeArray(((Array*) output_gradient)[i]);
    }
    
    pthread_mutex_unlock(&conv_layer->layer->mutex_layer);
    return inputs_gradients;
}


Array rotate_kernel_180(Array kernel){
    double **result = create_matrix_double(kernel->nRow, kernel->nCol);
    int i, j;

    for (i = 0; i < kernel->nRow; i++)
        for (j = 0; j < kernel->nCol; j++)
            result[i][j] = kernel->data[kernel->nRow - i - 1][kernel->nCol - j - 1];

    Array array = calloc(1 , sizeof(ArrayStr));
    array->nRow = kernel->nRow;
    array->nCol = kernel->nCol;
    array->data = result;
    return array;
}

//convolution sans rotation de filtre
Array cross_corolation(Array input , Array kernel){

    int height_output = input->nRow - kernel->nRow + 1; 
    int width_output = input->nCol - kernel->nCol + 1;

    double **result = create_matrix_double(height_output, width_output);
    int i, j, u, v;

    for (i = 0; i < height_output; i++)
    {
        for (j = 0; j < width_output; j++)
        {
            result[i][j] = 0;
            for (u = 0; u < kernel->nRow; u++)
                for (v = 0; v < kernel->nCol; v++)
                    result[i][j] += input->data[i + u][j + v] * kernel->data[u][v];
        }
    }
    Array array_result = malloc(sizeof(ArrayStr));
    array_result->nRow = height_output;
    array_result->nCol = width_output;
    array_result->data = result;
    return array_result;
}



Array full_convolotion_180(Array  input , Array kernel){

    Array kernel_180 = rotate_kernel_180(kernel);
    int height_output = input->nRow + kernel->nRow - 1;
    int width_output = input->nCol + kernel->nCol - 1;
    double **result = create_matrix_double(height_output, width_output);
    int i, j, u, v;

    for (i = 0; i < height_output; i++)
    {
        for (j = 0; j < width_output; j++)
        {
            result[i][j] = 0;
            for (u = 0; u < kernel_180->nRow; u++)
                for (v = 0; v < kernel_180->nCol; v++)
                    result[i][j] += (((i - u) < input->nRow && (i - u) >= 0 && (j - v) >= 0 && (j - v) < input->nCol) ? input->data[i - u][j - v] * kernel_180->data[u][v] : 0);
        }
    }

    Array array_result = malloc(sizeof(ArrayStr));
    array_result->nRow = height_output;
    array_result->nCol = width_output;
    array_result->data = result;

    freeArray(kernel_180);
   
    return array_result;
}

void free_convolution_layer(Convolution_Layer convolition_layer){
    int i = 0 , j = 0;
    for (i = 0; i < convolition_layer->depth; i++)
    {
        freeArray(convolition_layer->biases[i]);
    }

    for (i = 0; i < convolition_layer->input_shapes.depth; i++)
    {
        for (j = 0; j < convolition_layer->depth; j++)
        {
            freeArray(convolition_layer->kernels[i][j]);
        }
        
    }
    free(convolition_layer->layer);
    free(convolition_layer);
}