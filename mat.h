#ifndef MAT_H_
#define MAT_H_

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

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
#define diag_from mat_diag_from
#define vec mat_vec
#define vec_from mat_vec_from
#define free_mat mat_free_mat

/* DO NOT strip. May cause collisions */
// #define init mat_init
// #define size mat_size
// #define from mat_from
// #define at mat_at
// #define equals mat_equals
// #define equals_tol mat_equals_tol

/* Core operations are not stripped for readability */
// mat_add
// mat_radd
// mat_sub
// mat_rsub
// mat_mul
#endif // MAT_STRIP_PREFIX

#ifdef MAT_DOUBLE_PRECISION
  typedef double mat_elem_t;
#else
  typedef float mat_elem_t;
#endif // MAT_DOUBLE_PRECISION

#define identity eye

#ifndef MAT_DEFAULT_EPSILON
  #define MAT_DEFAULT_EPSILON 1e-6f
#endif

#ifndef MAT_LOG_LEVEL
  #define MAT_LOG_LEVEL 0
#endif

#if MAT_LOG_LEVEL > 0
  #include <stdio.h>
  #ifndef MAT_LOG_OUTPUT_ERR
    #define MAT_LOG_OUTPUT_ERR(msg) fprintf(stderr, "%s\n", msg)
  #endif
  #ifndef MAT_LOG_OUTPUT
    #define MAT_LOG_OUTPUT(msg) fprintf(stdout, "%s\n", msg)
  #endif
#endif

// ERROR outputs to stderr, WARN and INFO output to stdout
#if MAT_LOG_LEVEL >= 1
  #define MAT_LOG_ERROR(msg) MAT_LOG_OUTPUT_ERR("[ERROR] " msg)
#else
  #define MAT_LOG_ERROR(msg)
#endif

#if MAT_LOG_LEVEL >= 2
  #define MAT_LOG_WARN(msg) MAT_LOG_OUTPUT("[WARN] " msg)
#else
  #define MAT_LOG_WARN(msg)
#endif

#if MAT_LOG_LEVEL >= 3
  #define MAT_LOG_INFO(msg) MAT_LOG_OUTPUT("[INFO] " msg)
#else
  #define MAT_LOG_INFO(msg)
#endif


#ifndef MAT_ASSERT
#include <assert.h>
#define MAT_ASSERT(x) assert(x)
#endif

#ifndef MAT_ASSERT_MAT
#define MAT_ASSERT_MAT(m)                        \
  do {                                           \
    MAT_ASSERT((m) != NULL);                     \
    MAT_ASSERT((m)->data != NULL);               \
    MAT_ASSERT((m)->rows > 0 && (m)->cols > 0);  \
  } while(0)
#endif

#ifndef MAT_ASSERT_DIM
#define MAT_ASSERT_DIM(rows, cols)  \
  do {                              \
    MAT_ASSERT((rows) > 0);         \
    MAT_ASSERT((cols) > 0);         \
  } while(0)
#endif

#ifndef MAT_ASSERT_SQUARE
#define MAT_ASSERT_SQUARE(m)             \
  do {                                   \
    MAT_ASSERT_MAT(m);                   \
    MAT_ASSERT((m)->rows == (m)->cols);  \
  } while(0)
#endif

typedef struct {
  size_t rows;
  size_t cols;
  mat_elem_t *data;
} Mat;

typedef struct {
  size_t x;
  size_t y;
} MatSize;

typedef Mat Vec;

Mat *mat_empty(size_t rows, size_t cols);

Mat *mat_mat(size_t rows, size_t cols);

Vec *mat_vec(size_t dim);

Vec *mat_vec_from(size_t dim, mat_elem_t *values);

Mat *mat_from(size_t rows, size_t cols, mat_elem_t *values);

void mat_init(Mat *out, mat_elem_t *values);

void mat_free_mat(Mat *m);

Mat *mat_zeros(size_t rows, size_t cols);

Mat *mat_ones(size_t rows, size_t cols);

Mat *mat_eye(size_t dim);

void mat_print(Mat *m);

mat_elem_t mat_at(Mat *mat, size_t row, size_t col);

MatSize mat_size(Mat *m);

Mat *mat_rt(Mat *m);

void mat_t(Mat *out, Mat *m);

void mat_reshape(Mat *m, size_t rows, size_t cols);

Mat *mat_rreshape(Mat *m, size_t rows, size_t cols);

Vec *mat_diag(Mat *m);

Mat *mat_diag_from(size_t dim, mat_elem_t *values);

void mat_add(Mat *out, Mat *m1, Mat *m2);

Mat *mat_radd(Mat *m1, Mat *m2);

Mat *mat_rsub(Mat *m1, Mat *m2);

void mat_sub(Mat *out, Mat *m1, Mat *m2);

// TODO: implement
Mat *mat_add_many(size_t count, ...);

// TODO: implement
Mat *mat_mul(Mat *m1, Mat *m2);

// TODO: implement
Mat *mat_rmul(Mat *m1, Mat *m2);

bool mat_equals_tol(Mat *m1, Mat *m2, mat_elem_t epsilon);

bool mat_equals(Mat *m1, Mat *m2);

#endif // MAT_H_

#ifdef MAT_IMPLEMENTATION

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Construction functions  */

Mat *mat_empty(size_t rows, size_t cols) {
  MAT_ASSERT_DIM(rows, cols);
  
  Mat *mat = malloc(sizeof(Mat));
  mat->rows = rows;
  mat->cols = cols;
  mat->data = NULL;
  
  return mat;
}

Mat* mat_mat(size_t rows, size_t cols) {
  Mat *mat = mat_empty(rows, cols);
  mat->data = calloc(rows * cols, sizeof(mat_elem_t));
  
  return mat;
}

Vec *mat_vec(size_t dim) {
  Vec *vec = mat_empty(dim, 1);
  return vec;
}

Vec *mat_vec_from(size_t dim, mat_elem_t *values) {
  Vec *result = mat_from(dim, 1, values);
  
  return result;
}

Mat *mat_from(size_t rows, size_t cols, mat_elem_t *values) {
  MAT_ASSERT_DIM(rows, cols);
 
  Mat *result = mat_mat(rows, cols);
  mat_init(result, values);

  return result;
}

void mat_init(Mat *out, mat_elem_t *values) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT(values != NULL);

  for (size_t i = 0; i < out->rows; i++) {
    for (size_t j = 0; j < out->cols; j++) {
      out->data[i * out->cols + j] = values[i * out->cols + j];
    }
  }
}

void mat_free_mat(Mat *m) {
  MAT_ASSERT_MAT(m);
  
  free(m->data);
  free(m);
}

Mat *mat_zeros(size_t rows, size_t cols) { return mat_mat(rows, cols); }

Mat *mat_ones(size_t rows, size_t cols) {
  MAT_ASSERT_DIM(rows, cols);

  Mat *result = mat_mat(rows, cols);

  for (size_t i = 0; i < rows * cols; i++)
    result->data[i] = 1;

  return result;
}

Mat *mat_eye(size_t dim) {
  MAT_ASSERT(dim > 0);
  Mat *result = mat_mat(dim, dim);
    
  for (size_t i = 0; i < dim; i++) {
    result->data[i * dim + i] = 1;
  }

  return result;
}

// Mat utilities

void mat_print(Mat *mat) {
  MAT_ASSERT(mat != NULL);
  MAT_ASSERT(mat->data != NULL);

  printf("[");
  for (size_t i = 0; i < mat->rows; i++) {
    if (i > 0) printf(" ");
    for (size_t j = 0; j < mat->cols; j++) {
      printf("%g", mat->data[i * mat->cols + j]);
      if (j < mat->cols - 1) {
        printf(" ");
      }
    }
    if (i < mat->rows - 1) {
      printf(";\n");
    }
  }
  printf("]\n");
}

// Matrix operations

inline mat_elem_t mat_at(Mat *m, size_t row, size_t col) {
  MAT_ASSERT_MAT(m);
  return m->data[row * m->cols + col];
}

inline MatSize mat_size(Mat *m) {
  MatSize size = {m->rows, m->cols};
  return size;
}

Mat *mat_rt(Mat *m) {
  MAT_ASSERT_MAT(m);
  MAT_ASSERT_DIM(m->rows, m->cols);

  Mat *result = mat_mat(m->rows, m->cols);
  mat_t(result, m);

  return result;
}

// TODO: support transpose of rectangular matrices
//  `mat_pad` is needed
void mat_t(Mat *out, Mat *m) {
  MAT_ASSERT_SQUARE(out);
  MAT_ASSERT_SQUARE(m);

  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++) {
      out->data[j * m->rows + i] = m->data[i * m->cols + j];
    }
  }
}

void mat_reshape(Mat *out, size_t rows, size_t cols) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_DIM(rows, cols);
  MAT_ASSERT(out->rows * out->cols == rows * cols);

  out->rows = rows;
  out->cols = cols;
}

Mat *mat_rreshape(Mat *m, size_t rows, size_t cols) {
  MAT_ASSERT_MAT(m);
  MAT_ASSERT_DIM(rows, cols);

  Mat *result = mat_mat(rows, cols);

  result->rows = m->cols;
  result->cols = m->rows;

  return result;
}

Vec *mat_diag(Mat *m) {
  MAT_ASSERT_SQUARE(m);
  
  Vec *d = mat_vec(m->rows);

  for (size_t i = 0; i < d->rows; i++) {
    d->data[i] = m->data[i * d->rows + i];
  }

  return d;
}

Mat *mat_diag_from(size_t dim, mat_elem_t *values) {
  MAT_ASSERT(values != NULL);
  MAT_ASSERT(dim > 0);

  Mat *result = mat_mat(dim, dim);

  for (size_t i = 0; i < dim; i++) {
    result->data[i * dim + i] = values[i];
  }

  return result;
}

void mat_add(Mat *out, Mat *m1, Mat *m2) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(m1);
  MAT_ASSERT_MAT(m2);
  MAT_ASSERT(m1->rows == m2->rows);
  MAT_ASSERT(m1->cols == m2->cols);

  size_t rows = m1->rows;
  size_t cols = m1->cols;
 
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      out->data[i * cols + j] = mat_at(m1, i, j) + mat_at(m2, i, j);
    }
  }
}

Mat *mat_radd(Mat *m1, Mat *m2) {  
  Mat *out = mat_mat(m1->rows, m1->cols);
  mat_add(out, m1, m2);
  
  return out;
}

Mat *mat_rsub(Mat *m1, Mat *m2) {
  MAT_ASSERT_MAT(m1);
  MAT_ASSERT_MAT(m2);
 
  size_t rows = m1->rows;
  size_t cols = m1->cols;

  Mat *out = mat_mat(rows, cols);

  mat_sub(out, m1, m2);

  return out;
}

void mat_sub(Mat *out, Mat *m1, Mat *m2) {
  MAT_ASSERT_MAT(m1);
  MAT_ASSERT_MAT(m2);
  MAT_ASSERT(m1->rows == m2->rows);
  MAT_ASSERT(m1->cols == m2->cols);

  size_t rows = m1->rows;
  size_t cols = m1->cols;

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      out->data[i * cols + j] = mat_at(m1, i, j) - mat_at(m2, i, j);
    }
  }
}

bool mat_equals_tol(Mat *m1, Mat *m2, mat_elem_t epsilon) {
  MAT_ASSERT_MAT(m1);
  MAT_ASSERT_MAT(m2);

  if (m1->rows != m2->rows || m1->cols != m2->cols)
    return false;

  for (size_t i = 0; i < m1->rows * m1->cols; i++) {
    mat_elem_t diff = m1->data[i] - m2->data[i];
    if (diff < 0) diff = -diff;
    if (diff > epsilon)
      return false;
  }

  return true;
}

bool mat_equals(Mat *m1, Mat *m2) {
  return mat_equals_tol(m1, m2, MAT_DEFAULT_EPSILON);
}

#endif // MAT_IMPLEMENTATION
