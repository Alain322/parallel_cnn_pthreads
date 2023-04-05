#ifndef LOSSES_H
#define LOSSES_H
#include "../arrays/Arrayc.h"

double mse(Array y_pred, Array y_true);

Array mse_prime(Array y_pred, Array y_true);


#endif