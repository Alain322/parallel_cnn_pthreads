#ifndef CNN_H
#define CNN_H
#include "../layers/Layer.h"
#include "../arrays/Arrayc.h"

typedef struct params
{
	Layer *Network ;
	int network_len;
	double (*losse)(Array y_pred , Array y_true) ;
	Array (*losse_prime)(Array y_pred , Array y_true);
	int nbr_epoch ;
	Array **X_train ;
	Array *Y_train ;
	double learning_rate;
	int data_size;
	int thread_index;
	int nbr_thread;
}params_s , *params;

void train(Layer *Network , int network_len,
	double (*losse)(Array y_pred , Array y_true) , 
	Array (*losse_prime)(Array y_pred , Array y_true), 
	int nbr_epoch , Array **X_train , Array *Y_train , double learning_rate, int data_size);


void* parallele_train(void *arg);
void* parallele_train_bis(int NUM_THREADS , Layer *Network , int network_len,
	double (*losse)(Array y_pred , Array y_true) , Array (*losse_prime)(Array y_pred , Array y_true),
	int nbr_epoch , Array **X_train ,Array *Y_train , double learning_rate,int data_size);

Array predict(Layer *network , int network_len,  void* input);
params new_paramters(int nbr_thread , int index_thread,Layer *Network , int network_len,
	double (*losse)(Array y_pred , Array y_true) , Array (*losse_prime)(Array y_pred , Array y_true),
	int nbr_epoch , Array **X_train ,Array *Y_train , double learning_rate,int data_size);


#endif