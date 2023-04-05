#include <stdlib.h>
#include <math.h>

#include "../../headers/arrays/Arrayc.h"
#include "../../headers/arrays/base.h"

void freeArray(Array arg)
{
    if (arg != NULL)
    {
        free_matrix_double(arg->data, arg->nRow);
        free(arg);
    }
}

Array newArray(int nRow, int nCol, double **tab)
{
    Array rlt = NULL;
    if (nCol > 0 && nRow > 0 && tab != NULL)
    {
        Array rlt = malloc(sizeof(ArrayStr));
        if (rlt != NULL)
        {
            rlt->data = tab;
        }
    }
    return rlt;
}

Array valArray(int nRow, int nCol, double val)
{
    Array rlt = NULL;
    if (nCol > 0 && nRow > 0)
    {
        rlt = malloc(sizeof(ArrayStr));
        if (rlt != NULL)
        {
            rlt->data = create_matrix_double(nRow, nCol);
            if (rlt->data != NULL)
            {
                rlt->nRow = nRow;
                rlt->nCol = nCol;
                int i = 0, j = 0;
                for (i = 0; i < nRow; i++)
                {
                    for (j = 0; j < nCol; j++)
                    {
                        rlt->data[i][j] = val;
                    }
                }
            }
            else
            {
                free(rlt);
            }
        }
    }
    return rlt;
}

Array onesArray(int nRow, int nCol)
{
    return valArray(nRow, nCol, 1);
}

Array zerosArray(int nRow, int nCol)
{
    return valArray(nRow, nCol, 0);
}


Array identityArray(int n){
    Array rlt = NULL;
    if (n > 0)
    {
        rlt = malloc(sizeof(ArrayStr));
        if (rlt != NULL)
        {
            rlt->data = create_matrix_double(n, n);
            if (rlt->data != NULL)
            {
                rlt->nRow = n;
                rlt->nCol = n;
                int i = 0;
                for (i = 0; i < n; i++)
                {
                    rlt->data[i][i] = 1;
                }
            }
            else
            {
                free(rlt);
            }
        }
    }
    return rlt;
}


Array vectOpApplyArray(Array arg1 ,  Array vect , double (*op)(double, double)){
    int i = 0 , j = 0;

    Array rlt = NULL;
    //operation au niveau des colonnes
    if(vect->nRow == 1 && vect->nCol == arg1->nCol){   
        rlt = malloc(sizeof(ArrayStr));
        rlt->data = create_matrix_double(arg1->nRow , arg1->nCol);;
        rlt->nRow = arg1->nRow;
        rlt->nCol = arg1->nCol;
        for (j = 0; j < arg1->nCol; j++)
        {
            for (i = 0; i < arg1->nRow; i++)
            {
                rlt->data[i][j] = (*op)(arg1->data[i][j] , vect->data[0][j]);
                // printf("enter %f\n" , mat[i][j]);
            } 
        }
        // rlt = newArray(arg1->nRow , arg1->nCol , mat);
    }else if(vect->nCol == 1 && vect->nRow == arg1->nRow){
        rlt = malloc(sizeof(ArrayStr));
        rlt->data = create_matrix_double(arg1->nRow , arg1->nCol);;
        rlt->nRow = arg1->nRow;
        rlt->nCol = arg1->nCol;
        for (i = 0; i < arg1->nRow; i++)
        {
            for (j = 0; j < arg1->nCol; j++)
            {
                rlt->data[i][j] = (*op)(arg1->data[i][j] , vect->data[i][0]);
            } 
        }
    }else{
        freeArray(rlt);
    }

    return rlt;
   
}

boolean vectOpApplyArray_r(Array arg1 ,  Array vect , Array result , double (*op)(double, double)){
    if(result->nRow == arg1->nRow && result->nCol == arg1->nCol){
        int i = 0 , j = 0;
        //operation au niveau des colonnes
        if(vect->nRow == 1 && vect->nCol == arg1->nCol){
            for (j = 0; j < arg1->nCol; j++)
            {
                for (i = 0; i < arg1->nRow; i++)
                {
                    result->data[i][j] = (*op)(arg1->data[i][j] , vect->data[0][j]);
                } 
            }
            return True;
        }else if(vect->nCol == 1 && vect->nRow == arg1->nRow){
            for (i = 0; i < arg1->nRow; i++)
            {
                for (j = 0; j < arg1->nCol; j++)
                {
                    result->data[i][j] = (*op)(arg1->data[i][j] , vect->data[i][0]);
                } 
            }
        
            return True;
        }

        return False;
    }
    return False;
}

Array randomArray(int nRow, int nCol , float min , float max){
    srand(time(NULL));
    Array rlt = NULL;
    if (nCol > 0 && nRow > 0)
    {
        rlt = malloc(sizeof(ArrayStr));
        if (rlt != NULL)
        {
            rlt->data = create_matrix_double(nRow, nCol);
            if (rlt->data != NULL)
            {
                rlt->nRow = nRow;
                rlt->nCol = nCol;
                int i = 0, j = 0;
                for (i = 0; i < nRow; i++)
                {
                    for (j = 0; j < nCol; j++)
                    {
                        rlt->data[i][j] = float_rand(min , max);
                    }
                }
            }
            else
            {
                free(rlt);
            }
        }
    }
    return rlt;
}


Array constOpApplyArray(Array arg , double const_value, double (*op)(double, double)){
    Array rlt = NULL;
    if(arg != NULL){
        rlt = malloc(sizeof(ArrayStr));
        if (rlt != NULL)
        {
            rlt->data = create_matrix_double(arg->nRow, arg->nCol);
            if (rlt->data != NULL)
            {
                rlt->nRow = arg->nRow;
                rlt->nCol = arg->nCol;
                int i = 0, j = 0;
                for (i = 0; i < arg->nRow; i++)
                {
                    for (j = 0; j < arg->nCol; j++)
                    {
                        rlt->data[i][j] = (*op)(arg->data[i][j] , const_value);
                    }
                }
            }
            else
            {
                free(rlt);
            }
        }
    }
    return rlt;
}

boolean constOpApplyArray_r(Array arg , Array result, double const_value , double (*op)(double, double)){
    if(arg != NULL && result != NULL){
        if(arg->nRow == result->nRow && arg->nCol == result->nCol){
            int i = 0 , j = 0;
            for (i = 0; i < arg->nRow; i++)
            {
                for (j = 0; j < arg->nCol; j++)
                {
                    result->data[i][j] = (*op)(arg->data[i][j] , const_value);
                }
            }
            return True;
        }
    }

    return False;
}


void fprintfArray(FILE *stream, Array array, boolean complete)
{
    if (array != NULL)
    {
        fprintf(stream, "\n[Nrow : %d, Ncol : %d] \n\n", array->nRow, array->nCol);
        if (complete)
        {
            int i = 0, j = 0;
            for (i = 0; i < array->nRow; i++)
            {
                for (j = 0; j < array->nCol; j++)
                {
                    fprintf(stream, "%lf ", array->data[i][j]);
                }
                fprintf(stream, "\n");
            }
        }
    }
}

void printfArray(Array array, boolean complete)
{
    fprintfArray(stdout, array, complete);
}

Array op__Array(Array arg1, Array arg2, double (*f)(double, double))
{
    Array rlt = NULL;
    if (arg1 != NULL && arg2 != NULL)
    {
        rlt = zerosArray(arg1->nRow, arg1->nCol);
        if (rlt != NULL)
        {
            int i = 0, j = 0;
            for (i = 0; i < arg1->nRow; i++)
            {
                for (j = 0; j < arg1->nCol; j++)
                {
                    rlt->data[i][j] = (*f)(arg1->data[i][j], arg2->data[i][j]);
                }
            }
        }
    }
    return rlt;
}

boolean op__Array_r(Array arg1, Array arg2, Array result1, double (*f)(double, double))
{
    if (arg1 != NULL && arg2 != NULL)
    {
        Array rlt = op__Array(arg1, arg2, f);
        if (result1 != NULL)
        {
            free_matrix_double(result1->data, result1->nRow);
            result1->data = rlt->data;
            result1->nRow = rlt->nRow;
            result1->nCol = rlt->nCol;
            free(rlt);
        }
        else
        {
            result1 = rlt;
        }

        return True;
    }
    return False;
}

Array op_Array(Array arg, double (*f)(double))
{
    Array rlt = NULL;
    if (arg != NULL)
    {
        rlt = zerosArray(arg->nRow, arg->nCol);
        int i = 0, j = 0;
        for (i = 0; i < arg->nRow; i++)
        {
            for (j = 0; j < arg->nCol; j++)
            {
                rlt->data[i][j] = (*f)(arg->data[i][j]);
            }
        }
    }
    return rlt;
}

boolean op_Array_r(Array arg, Array result, double (*f)(double))
{
    if (arg != NULL)
    {
        Array rlt = op_Array(arg, f);
        if (result != NULL)
        {
            free_matrix_double(result->data, result->nRow);
            result->data = rlt->data;
            result->nRow = rlt->nRow;
            result->nCol = rlt->nCol;
            free(rlt);
        }
        else
        {
            result = rlt;
        }
        return True;
    }
    return False;
}

double plus_Array(double a, double b)
{
    return a + b;
}
double minus_Array(double a, double b)
{
    return a - b;
}
double mul_Array(double a, double b)
{
    return a * b;
}

double div_Array(double a , double b){
    if(b == 0){
        printf("division par zero detecte");
        exit(1);
    }
    return a/b;
}

Array plusArray(Array arg1, Array arg2)
{
    return op__Array(arg1, arg2, *plus_Array);
}

boolean plusArray_r(Array arg1, Array arg2, Array result)
{
    return op__Array_r(arg1, arg2, result, *plus_Array);
}

Array minusArray(Array arg1, Array arg2)
{
    return op__Array(arg1, arg2, *minus_Array);
}

boolean minusArray_r(Array arg1, Array arg2, Array result)
{
    return op__Array_r(arg1, arg2, result, *minus_Array);
}

Array mulArray(Array arg1, Array arg2)
{
    return op__Array(arg1, arg2, *plus_Array);
}

boolean mulArray_r(Array arg1, Array arg2, Array result)
{
    return op__Array_r(arg1, arg2, result, *mul_Array);
}

Array matMulArray(Array arg1, Array arg2)
{
    Array rlt = NULL;
    if (arg1 != NULL && arg2 != NULL)
    {
        rlt = zerosArray(arg1->nRow, arg2->nCol);
        if (rlt != NULL)
        {
            int i = 0, j = 0, k = 0;
            for (i = 0; i < arg1->nRow; i++)
                for (j = 0; j < arg2->nCol; j++)
                    for (k = 0; k < arg1->nCol; k++)
                        rlt->data[i][j] += arg1->data[i][k]*arg2->data[k][j];
        }
    }
    return rlt;
}

double sumElementArray(Array arg1){
    int i = 0 , j = 0;
    double result = 0;
    for (i = 0; i < arg1->nRow; i++)
    {
        for (j = 0; j < arg1->nCol; j++)
        {
            result += arg1->data[i][j];
        }
        
    }
    return result;
}

boolean matMulArray_r(Array arg1, Array arg2, Array result)
{
    if (arg1 != NULL && arg2 != NULL)
    {
        Array rlt = matMulArray(arg1, arg2);
        if (result != NULL)
        {
            free_matrix_double(result->data, result->nRow);
            result->data = rlt->data;
            result->nRow = rlt->nRow;
            result->nCol = rlt->nCol;
            free(rlt);
        }
        else
        {
            result = rlt;
        }

        return True;
    }
    return False;
}

Array transposeArray(Array arg){
    Array rlt = NULL;
    if (arg != NULL)
    {
        rlt = zerosArray(arg->nCol, arg->nRow);
        if (rlt != NULL)
        {
            int i = 0, j = 0;
            for (i = 0; i < arg->nCol; i++)
                for (j = 0; j < arg->nRow; j++)
                    rlt->data[i][j] = arg->data[j][i];
        }
    }
    return rlt;
}

boolean transposeArray_r(Array arg,Array result){
    if (arg != NULL)
    {
        Array rlt = transposeArray(arg);
        if (result != NULL)
        {
            free_matrix_double(result->data, result->nRow);
            result->data = rlt->data;
            result->nRow = rlt->nRow;
            result->nCol = rlt->nCol;
            free(rlt);
        }
        else
        {
            result = rlt;
        }
        
        return True;
    }
    return False;
}

Array plusArray_p(int n, int nRow, int nCol, ...)
{
    Array rlt = NULL;
    if (nRow > 0 && nCol > 0)
    {
        rlt = zerosArray(nRow, nCol);
        if (rlt != NULL)
        {
            va_list ap;
            int i = 0, j = 0;
            va_start(ap, nCol);

            Array currentArray = NULL;
            while (n > 0)
            {
                currentArray = va_arg(ap, Array);
                if (currentArray != NULL)
                {
                    for (i = 0; i < nRow; i++)
                    {
                        for (j = 0; j < nCol; j++)
                        {
                            rlt->data[i][j] += currentArray->data[i][j];
                        }
                    }
                }
                --n;
            }
            va_end(ap);
        }
    }
    return rlt;
}

boolean plusArray_pr(Array result, int n, int nRow, int nCol, ...)
{
    if (nRow > 0 && nCol > 0)
    {

        Array rlt = zerosArray(nRow, nCol);
        if (rlt == NULL)
        {
            return False;
        }

        va_list ap;
        va_start(ap, nCol);

        Array currentArray = NULL;
        int i = 0, j = 0;
        while (n > 0)
        {
            currentArray = va_arg(ap, Array);
            if (currentArray != NULL)
            {
                for (i = 0; i < nRow; i++)
                {
                    for (j = 0; j < nCol; j++)
                    {
                        rlt->data[i][j] += currentArray->data[i][j];
                    }
                }
            }
            --n;
        }
        va_end(ap);

        if (result != NULL)
        {
            freeArray(result);
        }
        result = rlt;
        return True;
    }
    return False;
}
