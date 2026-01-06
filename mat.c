#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#include "mat.h"

Mat *mat_empty(size_t rows, size_t cols) {
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

Mat *mat_zeros(size_t rows, size_t cols) { return mat_mat(rows, cols); }

Mat *mat_ones(size_t rows, size_t cols) {
  assert(rows > 0);
  assert(cols > 0);

  Mat *mat = mat_mat(rows, cols);

  for (size_t i = 0; i < rows * cols; i++)
    mat->data[i] = 1;

  return mat;
}

Mat *mat_eye(size_t dim) {
  assert(dim > 0);
  Mat *mat = mat_mat(dim, dim);
    
  for (size_t i = 0; i < dim; i++) {
    mat->data[i * dim + i] = 1;
  }

  return mat;
}

void mat_init(Mat *out, mat_elem_t *values) {
  assert(out != NULL);
  assert(out->data != NULL);
  assert(values != NULL);

  for (size_t i = 0; i < out->rows; i++) {
    for (size_t j = 0; j < out->cols; j++) {
      out->data[i * out->cols + j] = values[i * out->cols + j];
    }
  }
}

mat_elem_t mat_index(Mat *mat, size_t row, size_t col) {
  assert(mat != NULL);
  assert(mat->data != NULL);
  
  return mat->data[row * mat->cols + col];
}

Mat *mat_rt(Mat *m) {
  assert(m != NULL);
  assert(m->data != NULL);
  assert(m->rows > 0);
  assert(m->cols > 0);

  Mat *out = mat_mat(m->rows, m->cols);

  mat_t(out, m);

  return out;
}

void mat_free_mat(Mat *m) {
  free(m->data);
  free(m);
}

// TODO: support transpose of rectangular matrices
//  `mat_resize` is needed
void mat_t(Mat *out, Mat *m) {
  assert(out != NULL);
  assert(m != NULL);
  assert(m->data != NULL);
  assert(m->rows == m->cols);

  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++) {
      out->data[j * m->rows + i] = m->data[i * m->cols + j];
    }
  }
}

void mat_add(Mat *out, Mat *m1, Mat *m2) {
  assert(out != NULL); 
  assert(m1 != NULL);
  assert(m2 != NULL);
  assert(m1->rows == m2->rows);
  assert(m1->cols == m2->cols);

  size_t rows = m1->rows;
  size_t cols = m1->cols;
 
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      out->data[i * cols + j] = mat_index(m1, i, j) + mat_index(m2, i, j);
    }
  }
}

Mat *mat_radd(Mat *m1, Mat *m2) {  
  Mat *out = mat_mat(m1->rows, m1->cols);
  mat_add(out, m1, m2);
  
  return out;
}

Mat *mat_rsub(Mat *m1, Mat *m2) {
  assert(m1 != NULL);
  assert(m2 != NULL);

  size_t rows = m1->rows;
  size_t cols = m1->cols;

  Mat *out = mat_mat(rows, cols);

  mat_sub(out, m1, m2);

  return out;
}

void mat_sub(Mat *out, Mat *m1, Mat *m2) {
  assert(m1 != NULL);
  assert(m2 != NULL);
  assert(m1->rows == m2->rows);
  assert(m1->cols == m2->cols);

  size_t rows = m1->rows;
  size_t cols = m1->cols;
 
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      out->data[i * cols + j] = mat_index(m1, i, j) - mat_index(m2, i, j);
    }
  }
}

