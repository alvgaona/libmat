#ifndef MAT_H_
#define MAT_H_

#include <stddef.h>
#include <stdint.h>

#ifdef MAT_STRIP_PREFIX
#define empty mat_empty
#define mat mat_mat
#define zeros mat_zeros
#define ones mat_ones
#define eye mat_eye

#define t mat_t
#define rt mat_rt
#define reshape mat_reshape
#define rreshape mat_rreshape
#define diag mat_diag
#define diag_to_mat mat_diag_to_mat

/* These these functions may cause collision */
// #define init mat_init */
// #define index mat_index

/* Core operations are not stripped for readability  */
// mat_add
// mat_radd
// mat_add_many
// mat_sub
// mat_rsub
// mat_mul
// mat_rmul

#define free_mat mat_free_mat
#endif

#ifdef MAT_DOUBLE_PRECISION
  typedef double mat_elem_t;
#else
  typedef float mat_elem_t;
#endif

#define identity eye

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

Mat *mat_empty(size_t rows, size_t cols);

Mat *mat_mat(size_t rows, size_t cols);

void mat_free_mat(Mat *m);

Mat *mat_zeros(size_t rows, size_t cols);

Mat *mat_ones(size_t rows, size_t cols);

Mat *mat_eye(size_t dim);

void mat_init(Mat *out, mat_elem_t *values);

mat_elem_t mat_index(Mat *mat, size_t row, size_t col);

Mat *mat_rt(Mat *m);

void mat_t(Mat *out, Mat *m);

// TODO: implement
void mat_reshape(Mat *m, size_t rows, size_t cols);

// TODO: implement
Mat *mat_rreshape(Mat *m, size_t rows, size_t cols);

// TODO: implement
Diag *mat_diag(Mat *mat);

// TODO: implement
Mat *mat_diag_to_mat(Diag *diag);

// TODO: implement
Mat *mat_add_diag(Mat *m, Diag *d);

void mat_add(Mat *out, Mat *m1, Mat *m2);

Mat *mat_radd(Mat *m1, Mat *m2);

Mat *mat_rsub(Mat *m1, Mat *m2);

void mat_sub(Mat *out, Mat *m1, Mat *m2);

// TODO: implement
Mat *mat_mul(Mat *m1, Mat *m2);

#endif
