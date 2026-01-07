#ifndef MAT_H_
#define MAT_H_

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifndef MATDEF
#define MATDEF
#endif

// Overridable allocator macros
// Define these before including mat.h to use custom allocators
// (e.g., arenas)
#ifndef MAT_MALLOC
#define MAT_MALLOC(sz) malloc(sz)
#endif

#ifndef MAT_CALLOC
#define MAT_CALLOC(n, sz) calloc(n, sz)
#endif

#ifndef MAT_FREE
#define MAT_FREE(p) free(p)
#endif

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
#define hadamard mat_hadamard
#define rhadamard mat_rhadamard
#define add_scalar mat_add_scalar
#define radd_scalar mat_radd_scalar
#define add_many mat_add_many
#define radd_many mat_radd_many

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

#ifdef __cplusplus
extern "C" {
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

// Construction

MATDEF Mat *mat_empty(size_t rows, size_t cols);

MATDEF Mat *mat_mat(size_t rows, size_t cols);

MATDEF Vec *mat_vec(size_t dim);

MATDEF Vec *mat_vec_from(size_t dim, const mat_elem_t *values);

MATDEF Mat *mat_from(size_t rows, size_t cols, const mat_elem_t *values);

MATDEF void mat_init(Mat *out, const mat_elem_t *values);

MATDEF void mat_free_mat(Mat *m);

MATDEF Mat *mat_zeros(size_t rows, size_t cols);

MATDEF Mat *mat_ones(size_t rows, size_t cols);

MATDEF Mat *mat_eye(size_t dim);

// Utilities

MATDEF void mat_print(const Mat *m);

MATDEF Mat* mat_copy(const Mat *m);

MATDEF Mat* mat_deep_copy(const Mat *m);


// Matrix Operations

MATDEF mat_elem_t mat_at(const Mat *mat, size_t row, size_t col);

MATDEF MatSize mat_size(const Mat *m);

MATDEF Mat *mat_rt(const Mat *m);

MATDEF void mat_t(Mat *out, const Mat *m);

MATDEF void mat_reshape(Mat *m, size_t rows, size_t cols);

MATDEF Mat *mat_rreshape(const Mat *m, size_t rows, size_t cols);

MATDEF Vec *mat_diag(const Mat *m);

MATDEF Mat *mat_diag_from(size_t dim, const mat_elem_t *values);

MATDEF void mat_scale(Mat *m, mat_elem_t k);

MATDEF Mat *mat_rscale(const Mat *m, mat_elem_t k);

MATDEF void mat_add(Mat *out, const Mat *a, const Mat *b);

MATDEF Mat *mat_radd(const Mat *a, const Mat *b);

MATDEF void mat_sub(Mat *out, const Mat *a, const Mat *b);

MATDEF Mat *mat_rsub(const Mat *a, const Mat *b);

MATDEF void mat_add_many(Mat *out, size_t count, ...);

MATDEF Mat *mat_radd_many(size_t count, ...);

MATDEF void mat_mul(Mat *out, const Mat *a, const Mat *b);

MATDEF Mat *mat_rmul(const Mat *a, const Mat *b);

MATDEF void mat_hadamard(Mat *out, const Mat *a, const Mat *b);

MATDEF Mat *mat_rhadamard(const Mat *a, const Mat *b);

MATDEF bool mat_equals_tol(const Mat *a, const Mat *b, mat_elem_t epsilon);

MATDEF bool mat_equals(const Mat *a, const Mat *b);

MATDEF mat_elem_t mat_dot(const Vec *v1, const Vec *v2);

// TODO: implement
void mat_cross(Vec* out, const Vec *v1, const Vec *v2);

// TODO: implement
void mat_outer(Mat* out, const Vec *v1, const Vec *v2);

// TODO: implement
mat_elem_t mat_norm(const Mat *m);

// TODO: implement
mat_elem_t mat_normp(const Mat *m, size_t p);

// TODO: implement
mat_elem_t mat_norm_max(const Mat *m);

// TODO: implement
mat_elem_t mat_norm_f(const Mat *m);

#ifdef __cplusplus
}
#endif

#endif // MAT_H_

#ifdef MAT_IMPLEMENTATION

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

// Construction

MATDEF Mat *mat_empty(size_t rows, size_t cols) {
  MAT_ASSERT_DIM(rows, cols);

  Mat *mat = (Mat *)MAT_MALLOC(sizeof(Mat));
  mat->rows = rows;
  mat->cols = cols;
  mat->data = NULL;

  return mat;
}

MATDEF Mat* mat_mat(size_t rows, size_t cols) {
  Mat *mat = mat_empty(rows, cols);
  mat->data = (mat_elem_t *)MAT_CALLOC(rows * cols, sizeof(mat_elem_t));

  return mat;
}

MATDEF Vec *mat_vec(size_t dim) {
  Vec *vec = mat_mat(dim, 1);
  return vec;
}

MATDEF Vec *mat_vec_from(size_t dim, const mat_elem_t *values) {
  Vec *result = mat_from(dim, 1, values);

  return result;
}

MATDEF Mat *mat_from(size_t rows, size_t cols, const mat_elem_t *values) {
  MAT_ASSERT_DIM(rows, cols);

  Mat *result = mat_mat(rows, cols);
  mat_init(result, values);

  return result;
}

MATDEF void mat_init(Mat *out, const mat_elem_t *values) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT(values != NULL);

  for (size_t i = 0; i < out->rows; i++) {
    for (size_t j = 0; j < out->cols; j++) {
      out->data[i * out->cols + j] = values[i * out->cols + j];
    }
  }
}

MATDEF void mat_free_mat(Mat *m) {
  MAT_ASSERT_MAT(m);

  MAT_FREE(m->data);
  MAT_FREE(m);
}

MATDEF Mat *mat_zeros(size_t rows, size_t cols) { return mat_mat(rows, cols); }

MATDEF Mat *mat_ones(size_t rows, size_t cols) {
  MAT_ASSERT_DIM(rows, cols);

  Mat *result = mat_mat(rows, cols);

  for (size_t i = 0; i < rows * cols; i++)
    result->data[i] = 1;

  return result;
}

MATDEF Mat *mat_eye(size_t dim) {
  MAT_ASSERT(dim > 0);
  Mat *result = mat_mat(dim, dim);

  for (size_t i = 0; i < dim; i++) {
    result->data[i * dim + i] = 1;
  }

  return result;
}

// Utilities Impl

MATDEF void mat_print(const Mat *mat) {
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

MATDEF Mat* mat_copy(const Mat *m) {
  MAT_ASSERT_MAT(m);

  Mat *result = mat_mat(m->rows, m->cols);

  return result;
}

MATDEF Mat* mat_deep_copy(const Mat *m) {
  MAT_ASSERT_MAT(m);

  Mat *result = mat_copy(m);
  size_t len = m->rows * m->cols;
 
  memcpy(result->data, m->data, len * sizeof(mat_elem_t));

  return result;
}

// Operations Impl

MATDEF mat_elem_t mat_at(const Mat *m, size_t row, size_t col) {
  MAT_ASSERT_MAT(m);
  return m->data[row * m->cols + col];
}

MATDEF MatSize mat_size(const Mat *m) {
  MatSize size = {m->rows, m->cols};
  return size;
}

MATDEF Mat *mat_rt(const Mat *m) {
  MAT_ASSERT_MAT(m);

  Mat *result = mat_mat(m->cols, m->rows);
  mat_t(result, m);

  return result;
}

MATDEF void mat_t(Mat *out, const Mat *m) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(m);
  MAT_ASSERT(out->rows == m->cols && out->cols == m->rows);

  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++) {
      out->data[j * m->rows + i] = m->data[i * m->cols + j];
    }
  }
}

MATDEF void mat_reshape(Mat *out, size_t rows, size_t cols) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_DIM(rows, cols);
  MAT_ASSERT(out->rows * out->cols == rows * cols);

  out->rows = rows;
  out->cols = cols;
}

MATDEF Mat *mat_rreshape(const Mat *m, size_t rows, size_t cols) {
  MAT_ASSERT_MAT(m);
  MAT_ASSERT_DIM(rows, cols);

  Mat *result = mat_mat(rows, cols);

  result->rows = m->cols;
  result->cols = m->rows;

  return result;
}

MATDEF Vec *mat_diag(const Mat *m) {
  MAT_ASSERT_SQUARE(m);

  Vec *d = mat_vec(m->rows);

  for (size_t i = 0; i < d->rows; i++) {
    d->data[i] = m->data[i * d->rows + i];
  }

  return d;
}

MATDEF Mat *mat_diag_from(size_t dim, const mat_elem_t *values) {
  MAT_ASSERT(values != NULL);
  MAT_ASSERT(dim > 0);

  Mat *result = mat_mat(dim, dim);

  for (size_t i = 0; i < dim; i++) {
    result->data[i * dim + i] = values[i];
  }

  return result;
}

MATDEF void mat_scale(Mat *out, mat_elem_t k) {
  MAT_ASSERT_MAT(out);

  for (size_t i = 0; i < out->rows * out->cols; i++) {
    out->data[i] *= k;
  }
}

MATDEF Mat *mat_rscale(const Mat *m, mat_elem_t k) {
  Mat *result = mat_deep_copy(m);

  mat_scale(result, k);

  return result;
}

MATDEF void mat_add_scalar(Mat *out, mat_elem_t k) {
  MAT_ASSERT_MAT(out);

  for (size_t i = 0; i < out->rows * out->cols; i++) {
    out->data[i] += k;
  }
}

MATDEF Mat *mat_radd_scalar(const Mat *m, mat_elem_t k) {
  Mat *result = mat_deep_copy(m);

  mat_add_scalar(result, k);

  return result;
}

MATDEF void mat_add(Mat *out, const Mat *a, const Mat *b) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);
  MAT_ASSERT_MAT(b);
  MAT_ASSERT(a->rows == b->rows);
  MAT_ASSERT(a->cols == b->cols);

  size_t rows = a->rows;
  size_t cols = a->cols;

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      out->data[i * cols + j] = mat_at(a, i, j) + mat_at(b, i, j);
    }
  }
}

MATDEF Mat *mat_radd(const Mat *a, const Mat *b) {
  Mat *out = mat_mat(a->rows, a->cols);
  mat_add(out, a, b);

  return out;
}

MATDEF void mat_sub(Mat *out, const Mat *a, const Mat *b) {
  MAT_ASSERT_MAT(a);
  MAT_ASSERT_MAT(b);
  MAT_ASSERT(a->rows == b->rows);
  MAT_ASSERT(a->cols == b->cols);

  size_t rows = a->rows;
  size_t cols = a->cols;

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      out->data[i * cols + j] = mat_at(a, i, j) - mat_at(b, i, j);
    }
  }
}

MATDEF Mat *mat_rsub(const Mat *a, const Mat *b) {
  MAT_ASSERT_MAT(a);
  MAT_ASSERT_MAT(b);

  size_t rows = a->rows;
  size_t cols = a->cols;

  Mat *out = mat_mat(rows, cols);

  mat_sub(out, a, b);

  return out;
}

MATDEF bool mat_equals_tol(const Mat *a, const Mat *b, mat_elem_t epsilon) {
  MAT_ASSERT_MAT(a);
  MAT_ASSERT_MAT(b);

  if (a->rows != b->rows || a->cols != b->cols)
    return false;

  for (size_t i = 0; i < a->rows * a->cols; i++) {
    mat_elem_t diff = a->data[i] - b->data[i];
    if (diff < 0) diff = -diff;
    if (diff > epsilon)
      return false;
  }

  return true;
}

MATDEF bool mat_equals(const Mat *a, const Mat *b) {
  return mat_equals_tol(a, b, MAT_DEFAULT_EPSILON);
}

MATDEF void mat_mul(Mat *out, const Mat *a, const Mat *b) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);
  MAT_ASSERT_MAT(b);
  MAT_ASSERT(a->cols == b->rows);

  for (size_t i = 0; i < a->rows; i++) {
    for (size_t j = 0; j < b->cols; j++) {
      mat_elem_t sum = 0;
      for (size_t k = 0; k < a->cols; k++) {
        sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
      }
      out->data[i * out->cols + j] = sum;
    }
  }
}

MATDEF Mat *mat_rmul(const Mat *a, const Mat *b) {
  MAT_ASSERT_MAT(a);
  MAT_ASSERT_MAT(b);
  MAT_ASSERT(a->cols == b->rows);

  Mat* result = mat_mat(a->rows, b->cols);

  mat_mul(result, a, b);

  return result;
}

MATDEF void mat_hadamard(Mat *out, const Mat *a, const Mat *b) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);
  MAT_ASSERT_MAT(b);
  MAT_ASSERT(a->rows == b->rows);
  MAT_ASSERT(a->cols == b->cols);

  for (size_t i = 0; i < a->rows * a->cols; i++) {
    out->data[i] = a->data[i] * b->data[i];
  }
}

MATDEF Mat *mat_rhadamard(const Mat *a, const Mat *b) {
  MAT_ASSERT_MAT(a);
  MAT_ASSERT_MAT(b);
  MAT_ASSERT(a->rows == b->rows);
  MAT_ASSERT(a->cols == b->cols);

  Mat *result = mat_mat(a->rows, a->cols);
  mat_hadamard(result, a, b);

  return result;
}

MATDEF void mat_add_many(Mat *out, size_t count, ...) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT(count > 0);

  for (size_t i = 0; i < out->rows * out->cols; i++) {
    out->data[i] = 0;
  }

  va_list args;
  va_start(args, count);

  for (size_t i = 0; i < count; i++) {
    Mat *m = va_arg(args, Mat*);
    MAT_ASSERT_MAT(m);
    MAT_ASSERT(m->rows == out->rows && m->cols == out->cols);

    for (size_t j = 0; j < out->rows * out->cols; j++) {
      out->data[j] += m->data[j];
    }
  }

  va_end(args);
}

MATDEF Mat *mat_radd_many(size_t count, ...) {
  MAT_ASSERT(count > 0);

  va_list args;
  va_start(args, count);

  Mat *first = va_arg(args, Mat*);
  MAT_ASSERT_MAT(first);

  Mat *result = mat_deep_copy(first);

  for (size_t i = 1; i < count; i++) {
    Mat *m = va_arg(args, Mat*);
    MAT_ASSERT_MAT(m);
    MAT_ASSERT(m->rows == result->rows && m->cols == result->cols);

    for (size_t j = 0; j < result->rows * result->cols; j++) {
      result->data[j] += m->data[j];
    }
  }

  va_end(args);
  return result;
}

MATDEF mat_elem_t mat_dot(const Vec *v1, const Vec *v2) {
  MAT_ASSERT_MAT(v1);
  MAT_ASSERT_MAT(v2);

  mat_elem_t result = 0;
  for (size_t i = 0; i < v1->rows; i++) {
    result += v1->data[i] * v2->data[i];
  }

  return result;
}

#endif // MAT_IMPLEMENTATION
