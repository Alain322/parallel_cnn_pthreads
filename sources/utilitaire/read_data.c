#include <stdlib.h>
#include <stdio.h>
#include "../../headers/utilitaire/read_data.h"
#include "../../headers/arrays/Arrayc.h"



void** read_csv(char *filename, int rows, int cols , int  nbr_targets)
{
    FILE *file = fopen(filename, "r");

    int one_hot_len = nbr_targets;
    
    if(nbr_targets == 2){
        one_hot_len = 1;
    }

    int pos = 0;
    void **result;
    if (file != NULL)
    {
        result = calloc(2 , sizeof(void *));
        Array **X_train = calloc(rows , sizeof(Array*));
        
        
        Array *Y_train = calloc(rows , sizeof(Array));

        for (int i = 0; i < rows; i++)
        {
            double **mat_x = create_matrix_double(cols -1 , 1);
            
            double **mat_y = create_matrix_double(one_hot_len , 1);
            

            for (int j = 0; j < cols; j++)
            {
                if(j <= cols - 2){
                    fscanf(file , "%lf,", &mat_x[j][0]);
                    // printf("%f\n" , mat_x[j][0]);
                }else{
                    fscanf(file , "%d\n", &pos);
                    if(pos >= nbr_targets){
                        printf("erreur lors de la lecture de la cible elle ne doit pas etre superieur ou egal au nombre de cible");
                        free(result);
                        free_matrix_double(mat_x , cols - 1);
                        free_matrix_double(mat_y , one_hot_len);
                        exit(1);
                    }
                    // printf("target %d\n" , pos);
                    if(one_hot_len >= 2){
                        mat_y[pos][0] = 1;
                    }else{
                        mat_y[0][0] = pos;
                    }
                }
            }
            Array *observation_tab = calloc(1 , sizeof(Array));
            Array observation = calloc(1, sizeof(ArrayStr));
            observation->data = mat_x;
            observation->nRow = cols -1;
            observation->nCol = 1;
            observation_tab[0] = observation;

            Array target = calloc(1, sizeof(ArrayStr));
            target->data = mat_y;
            target->nRow = one_hot_len;
            target->nCol = 1;
            X_train[i] = observation_tab;
            Y_train[i] = target;
        }
        result[0] = X_train;
        result[1] = Y_train;
        return result;
    }
    return NULL;
}


