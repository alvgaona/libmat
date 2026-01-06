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

#define assert_mat(m) do { assert((m) != NULL); assert((m)->data != NULL); assert((m)->rows > 0 && (m)->cols > 0); } while(0)
#define assert_mat_dim(rows, cols) do { assert((rows) > 0); assert((cols) > 0); } while(0)
#define assert_mat_square(m) do { assert_mat(m); assert((m)->rows == (m)->cols); } while(0)

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

#endif
