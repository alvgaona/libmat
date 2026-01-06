#ifndef MAT_H_
#define MAT_H_

#include <stddef.h>
#include <stdint.h>

#ifdef MAT_DOUBLE_PRECISION
  typedef double mat_elem_t;
#else
  typedef float mat_elem_t;
#endif


typedef struct {
  size_t rows;
  size_t cols;
  mat_elem_t *data;
} Mat;

typedef struct {
  size_t rows;
  size_t cols;
  mat_elem_t *data;
} Diag;

Mat *empty_mat(size_t rows, size_t cols);

Mat *create_mat(size_t rows, size_t cols);

Mat *zeros(size_t rows, size_t cols);

Mat *ones(size_t rows, size_t cols);

Mat *identity(size_t dim);

void init_mat(Mat *out, mat_elem_t *values);

mat_elem_t index_mat(Mat *mat, size_t row, size_t col);

// TODO: implement
Mat *transpose(Mat *m);

// TODO: implement
Diag *diag(Mat *mat);

// TODO: implement
Mat *diag_to_mat(Diag *diag);

// TODO: implement
Mat *add_mat_diag(Mat *m, Diag *d);

Mat *add_mat_mat(Mat *m1, Mat *m2);

Mat *sub_mat_mat(Mat *m1, Mat *m2);

// TODO: implement
Mat *mul_mat_mat(Mat *m1, Mat *m2);

#endif
