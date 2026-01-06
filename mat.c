#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "mat.h"

/* Construct/Deconstruct functions  */

Mat *mat_empty(size_t rows, size_t cols) {
  assert_mat_dim(rows, cols);
  
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
  assert_mat_dim(rows, cols);
 
  Mat *result = mat_mat(rows, cols);
  mat_init(result, values);

  return result;
}

void mat_init(Mat *out, mat_elem_t *values) {
  assert_mat(out);
  assert(values != NULL);

  for (size_t i = 0; i < out->rows; i++) {
    for (size_t j = 0; j < out->cols; j++) {
      out->data[i * out->cols + j] = values[i * out->cols + j];
    }
  }
}

void mat_free_mat(Mat *m) {
  assert_mat(m);
  
  free(m->data);
  free(m);
}

Mat *mat_zeros(size_t rows, size_t cols) { return mat_mat(rows, cols); }

Mat *mat_ones(size_t rows, size_t cols) {
  assert_mat_dim(rows, cols);

  Mat *result = mat_mat(rows, cols);

  for (size_t i = 0; i < rows * cols; i++)
    result->data[i] = 1;

  return result;
}

Mat *mat_eye(size_t dim) {
  assert(dim > 0);
  Mat *result = mat_mat(dim, dim);
    
  for (size_t i = 0; i < dim; i++) {
    result->data[i * dim + i] = 1;
  }

  return result;
}

/* Matrix operations */

inline mat_elem_t mat_at(Mat *m, size_t row, size_t col) {
  assert_mat(m);
  return m->data[row * m->cols + col];
}

inline MatSize mat_size(Mat *m) {
  MatSize size = {m->rows, m->cols};
  return size;
}

Mat *mat_rt(Mat *m) {
  assert_mat(m);
  assert_mat_dim(m->rows, m->cols);

  Mat *result = mat_mat(m->rows, m->cols);
  mat_t(result, m);

  return result;
}

// TODO: support transpose of rectangular matrices
//  `mat_pad` is needed
void mat_t(Mat *out, Mat *m) {
  assert_mat_square(out);
  assert_mat_square(m);

  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++) {
      out->data[j * m->rows + i] = m->data[i * m->cols + j];
    }
  }
}

void mat_reshape(Mat *out, size_t rows, size_t cols) {
  assert_mat(out);
  assert_mat_dim(rows, cols);
  assert(out->rows * out->cols == rows * cols);

  out->rows = rows;
  out->cols = cols;
}

Mat *mat_rreshape(Mat *m, size_t rows, size_t cols) {
  assert(m != NULL);
  assert_mat_dim(rows, cols);

  Mat *result = mat_mat(rows, cols);

  result->rows = m->cols;
  result->cols = m->rows;

  return result;
}

Vec *mat_diag(Mat *m) {
  assert_mat_square(m);
  
  Vec *d = mat_vec(m->rows);

  for (size_t i = 0; i < d->rows; i++) {
    d->data[i] = m->data[i * d->rows + i];
  }

  return d;
}

Mat *mat_diag_from(size_t dim, mat_elem_t *values) {
  assert(values != NULL);
  assert(dim > 0);

  Mat *result = mat_mat(dim, dim);

  for (size_t i = 0; i < dim; i++) {
    result->data[i * dim + i] = values[i];
  }

  return result;
}

void mat_add(Mat *out, Mat *m1, Mat *m2) {
  assert_mat(out);
  assert_mat(m1);
  assert_mat(m2);
  assert(m1->rows == m2->rows);
  assert(m1->cols == m2->cols);

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
  assert_mat(m1);
  assert_mat(m2);
 
  size_t rows = m1->rows;
  size_t cols = m1->cols;

  Mat *out = mat_mat(rows, cols);

  mat_sub(out, m1, m2);

  return out;
}

void mat_sub(Mat *out, Mat *m1, Mat *m2) {
  assert_mat(m1);
  assert_mat(m2);
  assert(m1->rows == m2->rows);
  assert(m1->cols == m2->cols);

  size_t rows = m1->rows;
  size_t cols = m1->cols;

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      out->data[i * cols + j] = mat_at(m1, i, j) - mat_at(m2, i, j);
    }
  }
}

bool mat_equals_tol(Mat *m1, Mat *m2, mat_elem_t epsilon) {
  assert_mat(m1);
  assert_mat(m2);

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

