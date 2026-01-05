#ifndef UTIL_H_
#define UTIL_H_

#include "mat.h"

void print_mat(Mat *m);

size_t *flatten_mat(size_t **in, size_t rows, size_t cols);

#endif
