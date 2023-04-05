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



void free_data(void **data , int nbr_sample){
    if(data != NULL){
        int i = 0;
        Array **X_train = (Array**) data[0];
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

    int nbr_sample = 50;
    int nbr_target = 2;
    int nbr_feature = 4;
    int inputs_deep = 1;
    int nbr_epoch = 2;
    double learning_rate = 0.001;
    int nbr_output_layer = (nbr_target == 2) ? 1 : nbr_target;
    

    void **data = read_csv("./datasets/data.csv" , nbr_sample , nbr_feature + 1  , nbr_target);
    Array **X_train = data[0];
    Array *Y_train = data[1];

    
    Layer Network[] = {
        new_Convolution_Layer((Shapes){1 , nbr_feature , 1} , (Shapes){1 , 2 , 1} , 2 , 
        convolution_forward , convolution_backward)->layer,
        new_Reshape((Shapes){2 , 3 , 1} , reshape_forward , reshape_backward)->layer,
        new_dense(6 , 2 , -100 , 100 , dense_forward , dense_backward)->layer,
        new_Activation(2 , sigmoid , sigmoid_prime , activation_forward , activation_backward)->layer,
        new_dense(2 , nbr_output_layer , -100 , 100 , dense_forward , dense_backward)->layer,
        new_Activation(2 , sigmoid , sigmoid_prime , activation_forward , activation_backward)->layer,
        new_Softmax(softmax_forward , softmax_backward)->layer
    };
    int network_len = 7;



    top1();
    train(Network , network_len , mse , mse_prime , nbr_epoch , X_train , Y_train ,learning_rate, nbr_sample);
    top2();
    unsigned long temps = cpu_time();
	printf("\ntime sequentielle = %ld.%03ldms\n", temps/1000, temps%1000);

    top1();
    parallele_train_bis(NUM_THREADS , Network , network_len , mse , mse_prime , nbr_epoch , X_train , Y_train , learning_rate , nbr_sample);
    top2();
    temps = cpu_time();
	printf("\ntime parallele = %ld.%03ldms\n", temps/1000, temps%1000);
    


    return 0;
}