#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#include "mat.h"

Mat *empty_mat(size_t rows, size_t cols) {
  Mat *mat = malloc(sizeof(Mat));
  mat->rows = rows;
  mat->cols = cols;
  mat->data = NULL;

  return mat;
}

Mat* create_mat(size_t rows, size_t cols) {
  Mat *mat = empty_mat(rows, cols);
  mat->data = calloc(rows * cols, sizeof(mat_elem_t));

  return mat;
}

Mat *zeros(size_t rows, size_t cols) { return create_mat(rows, cols); }

Mat *ones(size_t rows, size_t cols) {
  assert(rows > 0);
  assert(cols > 0);

  Mat *mat = create_mat(rows, cols);

  for (size_t i = 0; i < rows * cols; i++)
    mat->data[i] = 1;

  return mat;
}

Mat *identity(size_t dim) {
  assert(dim > 0);
  Mat *mat = create_mat(dim, dim);
    
  for (size_t i = 0; i < dim; i++) {
    mat->data[i * dim + i] = 1;
  }

  return mat;
}

void init_mat(Mat *out, mat_elem_t *values) {
  assert(out != NULL);
  assert(out->data != NULL);
  assert(values != NULL);

  for (size_t i = 0; i < out->rows; i++) {
    for (size_t j = 0; j < out->cols; j++) {
      out->data[i * out->cols + j] = values[i * out->cols + j];
    }
  }
}

mat_elem_t index_mat(Mat *mat, size_t row, size_t col) {
  assert(mat != NULL);
  assert(mat->data != NULL);
  
  return mat->data[row * mat->cols + col];
}

Mat *rmat_t(Mat *m) {
  assert(m != NULL);
  assert(m->data != NULL);
  assert(m->rows > 0);
  assert(m->cols > 0);

  Mat *out = create_mat(m->rows, m->cols);

  mat_t(out, m);

  return out;
}

void free_mat(Mat *m) {
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

void add_mat_mat(Mat *out, Mat *m1, Mat *m2) {
  assert(out != NULL); 
  assert(m1 != NULL);
  assert(m2 != NULL);
  assert(m1->rows == m2->rows);
  assert(m1->cols == m2->cols);

  size_t rows = m1->rows;
  size_t cols = m1->cols;
 
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      out->data[i * cols + j] = index_mat(m1, i, j) + index_mat(m2, i, j);
    }
  }
}

Mat *radd_mat_mat(Mat *m1, Mat *m2) {  
  Mat *out = create_mat(m1->rows, m1->cols);
  add_mat_mat(out, m1, m2);
  
  return out;
}

Mat *rsub_mat_mat(Mat *m1, Mat *m2) {
  assert(m1 != NULL);
  assert(m2 != NULL);

  size_t rows = m1->rows;
  size_t cols = m1->cols;

  Mat *out = create_mat(rows, cols);

  sub_mat_mat(out, m1, m2);

  return out;
}

void sub_mat_mat(Mat *out, Mat *m1, Mat *m2) {
  assert(m1 != NULL);
  assert(m2 != NULL);
  assert(m1->rows == m2->rows);
  assert(m1->cols == m2->cols);

  size_t rows = m1->rows;
  size_t cols = m1->cols;
 
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      out->data[i * cols + j] = index_mat(m1, i, j) - index_mat(m2, i, j);
    }
  }
}

