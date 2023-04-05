#include "../../headers/layers/Activations.h"
#include "../../headers/arrays/Arrayc.h"
#include "../../headers/utilitaire/utilitaire.h"
#include <math.h>

#include <stdlib.h>

void* sigmoid(void *inputs){
    Array result = constOpApplyArray((Array) inputs , -1 , mul_Array);// -inputs
    op_Array_r(result , result , exp); //exp(-inputs)
    constOpApplyArray_r(result , result , 1 , plus_Array); // 1 + exp(-inputs)
    op_Array_r(result , result , inverse); // 1/1 + exp(-inputs)
    return result;
}


void* sigmoid_prime(void* inputs){
    Array S = sigmoid(inputs); // s
    Array moinS = constOpApplyArray(S , -1 , mul_Array); // -S
    constOpApplyArray_r(moinS , moinS , 1 , plus_Array); // 1 - S

    Array result = mulArray(S , moinS); // S *(1 - S)
    freeArray(moinS);
    freeArray(S);
    return result;
}