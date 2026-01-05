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
  mat->data = calloc(rows * cols, sizeof(size_t));

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

void init_mat(Mat *out, size_t *values) {
  assert(out != NULL);
  assert(out->data != NULL);
  assert(values != NULL);

  for (size_t i = 0; i < out->rows; i++) {
    for (size_t j = 0; j < out->cols; j++) {
      out->data[i * out->cols + j] = values[i * out->cols + j];
    }
  }
}

size_t index_mat(Mat* mat, size_t row, size_t col) {
  return mat->data[row * mat->cols + col];
}

Mat *add_mat_mat(Mat *m1, Mat *m2) {
  assert(m1 != NULL);
  assert(m2 != NULL);
  assert(m1->rows == m2->rows);
  assert(m2->cols == m2->cols);

  size_t rows = m1->rows;
  size_t cols = m1->cols;

  Mat *out = malloc(sizeof(Mat));
  out->rows = rows;
  out->cols = cols;
  out->data = calloc(rows * cols, sizeof(size_t));

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      out->data[i * cols + j] = index_mat(m1, i, j) + index_mat(m2, i, j);
    }
  }

  return out;
}

Mat *sub_mat_mat(Mat *m1, Mat *m2) {
  assert(m1 != NULL);
  assert(m2 != NULL);
  assert(m1->rows == m2->rows);
  assert(m2->cols == m2->cols);

  size_t rows = m1->rows;
  size_t cols = m1->cols;

  Mat *out = malloc(sizeof(Mat));
  out->rows = rows;
  out->cols = cols;
  out->data = calloc(rows * cols, sizeof(size_t));

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      out->data[i * cols + j] = index_mat(m1, i, j) - index_mat(m2, i, j);
    }
  }

  return out;
}

