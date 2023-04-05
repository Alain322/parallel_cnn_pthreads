#include <stdlib.h>
#include "../../headers/CNN/CNN.h"
#include "../../headers/arrays/Arrayc.h"
#include "../../headers/layers/Dense.h"
#include "../../headers/layers/Reshapes.h"



void train(Layer *Network , int network_len,
        double (*losse)(Array y_pred , Array y_true) , 
        Array (*losse_prime)(Array y_pred , Array y_true), 
        int nbr_epoch , Array **X_train , Array *Y_train ,double learning_rate, int data_len){
    
     int i = 0 , epoch = 0 , j = 0 ;
     Array output = NULL;
     void* grad = NULL;
    //  printf("enter\n");
     while (epoch < nbr_epoch)
     {
        double error = 0;
        for (i = 0; i < data_len; i++)
        {
            // printfArray(X_train[i][0] , True);
            //forward
            output = predict(Network , network_len , X_train[i]);
            // error
            // printfArray((Array) output , True);
            error += losse(output , Y_train[i]);
            //backward

            grad = losse_prime(output , Y_train[i]);
            for (j = network_len - 1 ; j >= 0; j--)
            {   
                grad = Network[j]->backward(Network[j]->child_layer , grad , learning_rate);
            }
           
        }
        
        error /= data_len;
        // printf("Epoch  => %d error => %f\n" , epoch, error);
        epoch++;
     }        
}


params new_paramters(int nbr_thread , int thread_index , Layer *Network , int network_len,
	double (*losse)(Array y_pred , Array y_true) , Array (*losse_prime)(Array y_pred , Array y_true),
	int nbr_epoch , Array **X_train ,Array *Y_train , double learning_rate,int data_size){

    params parameters = calloc(1 , sizeof(params_s));
    parameters->Network = Network;
    parameters->network_len = network_len;
    parameters->losse = losse;
    parameters->losse_prime = losse_prime;
    parameters->nbr_epoch = nbr_epoch;
    parameters->X_train = X_train;
    parameters->Y_train = Y_train;
    parameters->learning_rate = learning_rate;
    parameters->data_size = data_size;
    parameters->nbr_thread = nbr_thread;
    return parameters;
}

void* parallele_train(void *arg){
    params paramters = arg;
    int thread_index = paramters->thread_index;
    int i = 0 , j = 0;
    void *output;
    int epoch = 0;
    void *grad;
    // printf("enter %d\n" , paramters->nbr_epoch);
    while (epoch <  paramters->nbr_epoch)
    {
        double error = 0;
        // printf("%d\n" , paramters->data_size);
        for (i = 0; i < paramters->data_size; i++)
        {
            // printfArray(paramters->X_train[i][0] , True);
            if ((i % paramters->nbr_thread) == thread_index)
            {
                output = predict(paramters->Network , paramters->network_len , paramters->X_train[i]);

                error += paramters->losse(output , paramters->Y_train[i]);

                train(paramters->Network , paramters->network_len , paramters->losse , 
                paramters->losse_prime , paramters->nbr_epoch , paramters->X_train , paramters->Y_train , paramters->learning_rate , paramters->data_size);
                grad = paramters->losse_prime(output ,paramters->Y_train[i]);
                for (j = paramters->network_len - 1 ; j >= 0; j--)
                {   
                    grad = paramters->Network[j]->backward(paramters->Network[j]->child_layer , grad , paramters->learning_rate);
                
                }
            }
        }
        error /= paramters->data_size;
        printf("Epoch  => %d error => %f\n" , epoch, error);
        epoch++; 
    }

    return NULL;
}

void* parallele_train_bis(int NUM_THREADS , Layer *Network , int network_len,
	double (*losse)(Array y_pred , Array y_true) , Array (*losse_prime)(Array y_pred , Array y_true),
	int nbr_epoch , Array **X_train ,Array *Y_train , double learning_rate,int data_size){
    pthread_t thread[NUM_THREADS];
    
    int i = 0 , t = 0;
    int rc;
    params parameters = new_paramters(NUM_THREADS , t , Network , network_len , losse , losse_prime ,
    nbr_epoch , X_train , Y_train , learning_rate , data_size);
    for(t=0; t<NUM_THREADS; t++) {
		
		rc = pthread_create(&thread[t], NULL, parallele_train , (void *) parameters);
		if (rc) {
			printf("ERROR; return code from pthread_create() is %d\n", rc);
			exit(-1);
		}
	}

    return NULL;
}

//je dois faire les free au moment du backward
Array predict(Layer *network , int network_len, void *input){
    void* output = input;
    int i = 0;

    // for (i = 0; i < network_len; i++)
    // {
    //     output = network[i]->forward(network[i] , output);
    // }
    for (i = 0; i < network_len; i++)
    {
        output = network[i]->forward(network[i]->child_layer , output);
    }
    
    return output;
}