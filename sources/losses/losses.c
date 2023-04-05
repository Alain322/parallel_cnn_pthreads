#include <stdlib.h>
#include "../../headers/losses/losses.h"
#include "../../headers/utilitaire/utilitaire.h"


double mse(Array y_pred, Array y_true){
    Array temp = minusArray(y_pred , y_true);
    op_Array_r(temp , temp , pow2);
    double value = sumElementArray(temp);
    value /= (y_true->nRow * y_true->nCol);
    freeArray(temp); 
    return value;
}


Array mse_prime(Array y_pred, Array y_true){
    Array temp = minusArray(y_pred , y_true);
    constOpApplyArray_r(temp , temp , 2 , mul_Array);
    int size = y_true->nRow * y_true->nCol;
    constOpApplyArray_r(temp , temp , size , div_Array);
    return temp;
}

