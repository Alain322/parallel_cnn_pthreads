#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <limits.h>

#include "headers/arrays/Arrayc.h"
#include "headers/layers/Dense.h"
#include "headers/layers/softmax.h"
#include "headers/losses/losses.h"
#include "headers/layers/Activation.h"
#include "headers/layers/Activations.h"
#include "headers/utilitaire/read_data.h"
#include "headers/layers/Convolution_Layer.h"
#include "headers/CNN/CNN.h"
#include "headers/layers/Reshapes.h"


int NUM_THREADS = 8;

typedef struct timezone timezone_t;
typedef struct timeval timeval_t;

timeval_t t1, t2;
timezone_t tz;


static struct timeval _t1, _t2;
static struct timezone _tz;
timeval_t t1, t2;
timezone_t tz;

static unsigned long _temps_residuel = 0;
#define top1() gettimeofday(&_t1, &_tz)
#define top2() gettimeofday(&_t2, &_tz)

void init_cpu_time(void)
{
   top1(); top2();
   _temps_residuel = 1000000L * _t2.tv_sec + _t2.tv_usec -
                     (1000000L * _t1.tv_sec + _t1.tv_usec );
}

unsigned long cpu_time(void) /* retourne des microsecondes */
{
   return 1000000L * _t2.tv_sec + _t2.tv_usec -
           (1000000L * _t1.tv_sec + _t1.tv_usec ) - _temps_residuel;
}

// void* generate_data(int nbr_sample , int nbr_feature , int nbr_target);

// void* generate_data(int nbr_sample , int nbr_feature , int nbr_target){
//     int i = 0;
//     Array *X_train = calloc(nbr_sample , sizeof(ArrayStr));
//     Array *Y_train = calloc(nbr_sample , sizeof(ArrayStr));
//     void **data = calloc(2 , sizeof(void*));

//     for (i = 0; i < nbr_sample; i++)
//     {
//         // X_train[i] = valArray(nbr_feature , 1 , 1 * i);
//         // if(i%2 == 0){
//         //     Y_train[i] = valArray(nbr_target , 1 , 1);
//         // }else{
//         //     Y_train[i] = valArray(nbr_target , 1 , 0);
//         // }

//         X_train[i] = randomArray(nbr_feature , 1 , 0 , 10);
//         Y_train[i] = randomArray(nbr_target , 1 , 0 , 1);
//     }
//     data[0] = X_train;
//     data[1] = Y_train;
//     return data;
// }

void free_data(void **data , int nbr_sample){
    if(data != NULL){
        int i = 0;
        Array **X_train = (Array*) data[0];
        Array *Y_train = (Array*) data[1];

        for (i = 0; i < nbr_sample; i++)
        {
            freeArray(X_train[i][0]);
            freeArray(Y_train[i]);
        }
    }
}



int main(int argc, char const *argv[]){

    int i = 0 , j = 0;
    if (argc != 2)
	{
		printf("<thread_number>\n");
		return 1;
	}else{
		NUM_THREADS = atoi(argv[1]);
	}	
    pthread_t thread[NUM_THREADS];


    // srand(time(NULL));
    // int i = 0 ,  j = 0;

    // convolution layer initialisation


    // for (i = 0; i < 3 ; i++)
    // {
    //     for (j = 0; j < 2; j++)
    //     {
    //         printfArray(layer->kernels[i][j] , True);
    //     }
        
    // }
    // int input_depth = 2;
    // int heigth = 3;
    // int width = 2;
    // Array *inputs = calloc(input_depth , sizeof(Array));
    // for (i = 0; i < input_depth; i++)
    // {
    //     inputs[i] = randomArray(heigth , width , 0 , 10);
    // }
    

    // Layer layer = new_Convolution_Layer((Shapes){1 , heigth , width} , (Shapes){1 , 2 , 1} , 2 , convolution_backward , convolution_backward)->layer;

    // // //  forward pass in CNN

    // Array *outputs_gradient = convolution_forward(layer->child_layer , inputs);
    // for (i = 0; i < 2; i++)
    // {
    //     printfArray(outputs_gradient[i], True);
    // }


    // Array *inputs_gradient = convolution_backward(layer->child_layer , outputs_gradient , 0.1);

    // for (i = 0; i < 3; i++)
    // {
    //     printfArray(inputs_gradient[i] , True);
    // }


    
    // Array tab = randomArray(10 , 10 , 0 , 10);
    // Array kernel = randomArray(2 , 2 , 0, 5);
    // printfArray(tab , True);
    // printfArray(kernel , True);
    // printfArray(cross_corolation(tab , kernel) , True);

    // for (i = 0; i < 2; i++)
    // {
    //     printfArray(outputs[i], True);
    // }
    // free_convolution_layer(layer);

    // Layer layer = new_Reshape((Shapes){input_depth , heigth , width} , reshape_forward , reshape_backward)->layer;
    // for (i = 0; i < input_depth; i++)
    // {
    //     printfArray(inputs[i] , True);
    // }
    
    // Array flat = layer->forward(layer->child_layer , inputs);

    // printf("-----------------------------deflat---------------------------------------\n");
    // Array *deflat = layer->backward(layer->child_layer , flat , 0.1);

    // for (i = 0; i < input_depth; i++)
    // {
    //     printfArray(deflat[i] , True);
    // }
    

    // int nbr_sample = 50;
    // int nbr_target = 2;
    // int nbr_feature = 4;
    // int inputs_deep = 1;
    // int nbr_output_layer = (nbr_target == 2) ? 1 : nbr_target;
    
    // // // void **data = generate_data(nbr_sample , nbr_feature , nbr_target);
    // // // Array *X_train = data[0];
    // // // Array *Y_train = data[1];
    // // int i = 0;
    // void **data = read_csv("/home/dimitri/CNN_C/datasets/data.csv" , nbr_sample , nbr_feature + 1  , nbr_target);
    // Array **X_train = data[0];
    // Array *Y_train = data[1];
    // for (i = 0; i < nbr_sample; i++)
    // {
    //     printfArray(X_train[i][0] , True);
    //     printfArray(Y_train[i] , True);
    // }
    
    // // // free_data(data , 5);

    // // // Array inputs = randomArray(nbr_feature , 1 , 10 , 100);
    // // // // Array y = randomArray(3 , 1 , 0 , 1);
    
    // Layer Network[] = {
    //     new_Convolution_Layer((Shapes){1 , nbr_feature , 1} , (Shapes){1 , 2 , 1} , 2 , 
    //     convolution_forward , convolution_backward)->layer,
    //     // new_Activation(2 , sigmoid , sigmoid_prime , activation_forward_depth , activation_backward_depth)->layer,
    //     new_Reshape((Shapes){2 , 3 , 1} , reshape_forward , reshape_backward)->layer,
    //     new_dense(6 , 2 , -100 , 100 , dense_forward , dense_backward)->layer,
    //     new_Activation(2 , sigmoid , sigmoid_prime , activation_forward , activation_backward)->layer,
    //     new_dense(2 , nbr_output_layer , -100 , 100 , dense_forward , dense_backward)->layer,
    //     new_Activation(2 , sigmoid , sigmoid_prime , activation_forward , activation_backward)->layer,
    //     new_Softmax(softmax_forward , softmax_backward)->layer
    // };
    // int network_len = 7;


    // Layer layer = new_Convolution_Layer((Shapes){1 , nbr_feature , 1} , (Shapes){1 , 2 , 1} , 2 , convolution_forward , convolution_backward)->layer;
    // Array *outlayer = layer->forward(layer->child_layer , X_train[0]);
    // for (i = 0; i < ((Convolution_Layer) layer->child_layer)->output_shapes.depth; i++)
    // {
    //     printfArray(outlayer[i] , True);
    // }
    // Layer layer =  new_Activation(input_depth , sigmoid , sigmoid_prime , activation_forward_depth , activation_backward_depth)->layer;
    // Array *toto = layer->forward(layer->child_layer ,inputs);
    // for (i = 0; i < 2; i++)
    // {
    //     printfArray(toto[i] , True);
    // }
    
    // predict(Network , network_len , X_train[0]);

    // top1();
    // train(Network , network_len , mse , mse_prime , 2 , X_train , Y_train ,0.1, nbr_sample);
    // top2();
    // unsigned long temps = cpu_time();
	// printf("\ntime sequentiel = %ld.%03ldms\n", temps/1000, temps%1000);


    // Layer s = new_Softmax(softmax_forward , softmax_backward)->layer;
    // Array T = s->forward(s->child_layer , inputs);
    // printfArray(T , True);
    // Array prim = mse_prime(T , randomArray(2 , 1 , 0 , 1));

    // Array a = softmax_backward(s->child_layer , prim , 0.1);
    // printfArray(predict(Network , network_len , inputs) , True);
    // free_data(data , nbr_sample);
    // Array pred = predict(Network , network_len , inputs);
    // Array grad = mse_prime(pred , y);
    // for (i = network_len - 1 ; i <= 0; i--)
    // {
    //     grad = Network[i]->backward(Network[i] , grad , 0.1);
    // }
            
    
    // printfArray(pred , True);

    // printf("%f" , exp2(5));

    // srand(time(NULL));
    // int i = 0;
    // int *a = toto();
    // for (i = 0; i < 3; i++)
    // {
    //     printf("%d\n", a[i]);
    // }

    // Array array = valArray(2 , 2 , 7);
    // Array temp = constOpApplyArray(array , 2 , plus_Array);
    // // constOpApplyArray_r(array , array , 2 , plus_Array);
    // printfArray(temp , True);
    // printfArray(array , True);

    // printfArray(inputs , True);
    // printf("----------------------------------------\n");
    // Array vect = randomArray(3 , 1 , 3 , 7);
    // printfArray(vect , True);
    // printf("----------------------------------------\n");
    // Array temp = vectOpApplyArray(inputs , vect , plus_Array);
    // printfArray(temp , True);
    
    // Dense dense = new_dense(3 , 2 , 0 , 10 , dense_forward , dense_backward);
    // printfArray(dense->weights , True);
    
    // printfArray(dense->layer->forward(dense , inputs) , True);

    // Array temp = dense->layer->forward(dense , inputs);

    // Softmax softmax = new_Softmax(softmax_forward , softmax_backward);

    // Array temp2 = softmax->layer->forward(softmax , temp);

    // Array y_true = randomArray(2 , 1 , 0 , 1);
    // printfArray(temp , True);
    // Activation activation = new_Activation(sigmoid , sigmoid_prime , activation_forward , activation_backward);
    
    // printfArray(activation->layer->forward(activation , temp) ,True);
    // double e = mse(temp2 , y_true);
    // printfArray(temp2 , True);

    // printfArray(y_true , True);
    // printf("eeee %f\n" , e);
    // Array test1 = randomArray(5 , 3 , 0 , 10);
    // Array test2 = randomArray(3 , 2 , 0 , 10);

    // mulArray(test1 , test2);
    // freeArray(test1);
    // freeArray(test2);
    // freeArray(inputs);
    // free_dense_layer(dense);
    // printf("%f" , pow(2 , -1));

    return 0;
}