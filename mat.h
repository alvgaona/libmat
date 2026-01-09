#ifndef MAT_H_
#define MAT_H_

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// ARM NEON detection
#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
  #define MAT_HAS_ARM_NEON
  #include <arm_neon.h>
#endif

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

#define mat mat_mat
#define empty mat_empty
#define zeros mat_zeros
#define ones mat_ones
#define eye mat_eye
#define t mat_t
#define rt mat_rt
#define reshape mat_reshape
#define rreshape mat_rreshape
#define diag mat_diag
#define diag_from mat_diag_from
#define vec_from mat_vec_from
#define free_mat mat_free_mat
#define hadamard mat_hadamard
#define rhadamard mat_rhadamard
#define add_scalar mat_add_scalar
#define radd_scalar mat_radd_scalar
#define add_many mat_add_many
#define radd_many mat_radd_many

#endif // MAT_STRIP_PREFIX

// Initialization macros
#define mat_new(cols, ...) \
  mat_from( \
    sizeof((mat_elem_t[][cols])__VA_ARGS__) / sizeof(mat_elem_t[cols]), \
    cols, \
    (mat_elem_t*)((mat_elem_t[][cols])__VA_ARGS__) \
  )

#define mat_set(out, ...) \
  mat_init(out, (mat_elem_t[])__VA_ARGS__)

#define mat_vnew(...) \
  mat_vec_from( \
    sizeof((mat_elem_t[])__VA_ARGS__) / sizeof(mat_elem_t), \
    (mat_elem_t[])__VA_ARGS__ \
  )

#define mat_rnew(...) \
  mat_from( \
    1, \
    sizeof((mat_elem_t[])__VA_ARGS__) / sizeof(mat_elem_t), \
    (mat_elem_t[])__VA_ARGS__ \
  )

// Element type (float or double precision)
#ifdef MAT_DOUBLE_PRECISION
  typedef double mat_elem_t;
  #ifndef MAT_DEFAULT_EPSILON
    #define MAT_DEFAULT_EPSILON 1e-9
  #endif
#else
  typedef float mat_elem_t;
  #ifndef MAT_DEFAULT_EPSILON
    #define MAT_DEFAULT_EPSILON 1e-6f
  #endif
#endif

// NEON SIMD macros (only defined when targeting ARM with NEON)
#ifdef MAT_HAS_ARM_NEON
  #ifdef MAT_DOUBLE_PRECISION
    // NEON double precision (2 doubles per 128-bit register)
    #define MAT_NEON_TYPE     float64x2_t
    #define MAT_NEON_UTYPE    uint64x2_t
    #define MAT_NEON_WIDTH    2
    #define MAT_NEON_LOAD     vld1q_f64
    #define MAT_NEON_STORE    vst1q_f64
    #define MAT_NEON_DUP      vdupq_n_f64
    #define MAT_NEON_DUP_U    vdupq_n_u64
    #define MAT_NEON_FMA      vfmaq_f64
    #define MAT_NEON_ADD      vaddq_f64
    #define MAT_NEON_ADDV     vaddvq_f64
    #define MAT_NEON_ABS      vabsq_f64
    #define MAT_NEON_MAX      vmaxq_f64
    #define MAT_NEON_MAXV     vmaxvq_f64
    #define MAT_NEON_ABD      vabdq_f64
    #define MAT_NEON_CGT      vcgtq_f64
    #define MAT_NEON_CEQ      vceqq_f64
    #define MAT_NEON_ORR_U    vorrq_u64
    #define MAT_NEON_AND_U    vandq_u64
    #define MAT_NEON_MVN_U(x) veorq_u64(x, vdupq_n_u64(0xFFFFFFFFFFFFFFFFULL))
    #define MAT_NEON_MAXV_U(x) (vgetq_lane_u64(x, 0) | vgetq_lane_u64(x, 1))
    #define MAT_NEON_ADDV_U(x) (vgetq_lane_u64(x, 0) + vgetq_lane_u64(x, 1))
    #define MAT_NEON_ADD_U    vaddq_u64
  #else
    // NEON single precision (4 floats per 128-bit register)
    #define MAT_NEON_TYPE     float32x4_t
    #define MAT_NEON_UTYPE    uint32x4_t
    #define MAT_NEON_WIDTH    4
    #define MAT_NEON_LOAD     vld1q_f32
    #define MAT_NEON_STORE    vst1q_f32
    #define MAT_NEON_DUP      vdupq_n_f32
    #define MAT_NEON_DUP_U    vdupq_n_u32
    #define MAT_NEON_FMA      vfmaq_f32
    #define MAT_NEON_ADD      vaddq_f32
    #define MAT_NEON_ADDV     vaddvq_f32
    #define MAT_NEON_ABS      vabsq_f32
    #define MAT_NEON_MAX      vmaxq_f32
    #define MAT_NEON_MAXV     vmaxvq_f32
    #define MAT_NEON_ABD      vabdq_f32
    #define MAT_NEON_CGT      vcgtq_f32
    #define MAT_NEON_CEQ      vceqq_f32
    #define MAT_NEON_ORR_U    vorrq_u32
    #define MAT_NEON_AND_U    vandq_u32
    #define MAT_NEON_MVN_U(x) vmvnq_u32(x)
    #define MAT_NEON_MAXV_U(x) vmaxvq_u32(x)
    #define MAT_NEON_ADDV_U(x) vaddvq_u32(x)
    #define MAT_NEON_ADD_U    vaddq_u32
  #endif
#endif // __ARM_NEON

// Control visibility of internal implementations
#ifdef MAT_EXPOSE_INTERNALS
  #define MAT_INTERNAL_STATIC
#else
  #define MAT_INTERNAL_STATIC static
#endif

#define identity eye

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

// Construction & Memory

MATDEF Mat *mat_empty(size_t rows, size_t cols);
MATDEF Mat *mat_mat(size_t rows, size_t cols);
MATDEF Mat *mat_from(size_t rows, size_t cols, const mat_elem_t *values);
MATDEF void mat_init(Mat *out, const mat_elem_t *values);
MATDEF void mat_free_mat(Mat *m);

MATDEF Mat *mat_zeros(size_t rows, size_t cols);
MATDEF Mat *mat_ones(size_t rows, size_t cols);
MATDEF Mat *mat_eye(size_t dim);

MATDEF Vec *mat_vec(size_t dim);
MATDEF Vec *mat_vec_from(size_t dim, const mat_elem_t *values);

MATDEF Mat *mat_copy(const Mat *m);
MATDEF Mat *mat_deep_copy(const Mat *m);

// Accessors & Info

MATDEF mat_elem_t mat_at(const Mat *mat, size_t row, size_t col);
MATDEF MatSize mat_size(const Mat *m);
MATDEF void mat_print(const Mat *m);

// Comparison

MATDEF bool mat_equals(const Mat *a, const Mat *b);
MATDEF bool mat_equals_tol(const Mat *a, const Mat *b, mat_elem_t epsilon);

// Element-wise Unary
// TODO: mat_sqrt, mat_exp, mat_log, mat_pow, mat_clip

MATDEF void mat_abs(Mat *out, const Mat *a);

// Scalar Operations

MATDEF void mat_scale(Mat *out, mat_elem_t k);
MATDEF Mat *mat_rscale(const Mat *m, mat_elem_t k);
MATDEF void mat_add_scalar(Mat *out, mat_elem_t k);
MATDEF Mat *mat_radd_scalar(const Mat *m, mat_elem_t k);

// Matrix Arithmetic

MATDEF void mat_add(Mat *out, const Mat *a, const Mat *b);
MATDEF Mat *mat_radd(const Mat *a, const Mat *b);
MATDEF void mat_sub(Mat *out, const Mat *a, const Mat *b);
MATDEF Mat *mat_rsub(const Mat *a, const Mat *b);
MATDEF void mat_add_many(Mat *out, size_t count, ...);
MATDEF Mat *mat_radd_many(size_t count, ...);

// Matrix Products
// TODO: mat_cross, mat_outer

MATDEF void mat_mul(Mat *out, const Mat *a, const Mat *b);
MATDEF Mat *mat_rmul(const Mat *a, const Mat *b);
MATDEF void mat_hadamard(Mat *out, const Mat *a, const Mat *b);
MATDEF Mat *mat_rhadamard(const Mat *a, const Mat *b);
MATDEF mat_elem_t mat_dot(const Vec *v1, const Vec *v2);
MATDEF void mat_cross(Vec *out, const Vec *v1, const Vec *v2);
MATDEF void mat_outer(Mat *out, const Vec *v1, const Vec *v2);

// Structure Operations
// TODO: mat_hcat, mat_vcat, mat_slice, mat_row, mat_col

MATDEF void mat_t(Mat *out, const Mat *m);
MATDEF Mat *mat_rt(const Mat *m);
MATDEF void mat_reshape(Mat *out, size_t rows, size_t cols);
MATDEF Mat *mat_rreshape(const Mat *m, size_t rows, size_t cols);
MATDEF void mat_hcat(Mat *out, const Mat *a, const Mat *b);
MATDEF void mat_vcat(Mat *out, const Mat *a, const Mat *b);
MATDEF void mat_slice(Mat *out, const Mat *a, const Mat *b);

// Diagonal Operations

MATDEF Vec *mat_diag(const Mat *m);
MATDEF Mat *mat_diag_from(size_t dim, const mat_elem_t *values);

// Reduction Operations
// TODO: mat_sum, mat_mean, mat_min, mat_max
// TODO: mat_sum_rows, mat_sum_cols, mat_argmin, mat_argmax, mat_var, mat_std

// Norms

MATDEF mat_elem_t mat_norm(const Mat *a, size_t p);
MATDEF mat_elem_t mat_norm2(const Mat *a);
MATDEF mat_elem_t mat_norm_max(const Mat *a);
MATDEF mat_elem_t mat_norm_fro(const Mat *a);

// Matrix Properties
// TODO: mat_det, mat_rank, mat_cond, mat_inv, mat_pinv

MATDEF mat_elem_t mat_trace(const Mat *a);
MATDEF mat_elem_t mat_det(const Mat *a);
MATDEF mat_elem_t mat_rank(const Mat *a);
MATDEF mat_elem_t mat_cond(const Mat *a);
MATDEF mat_elem_t mat_nnz(const Mat *a);

// Decomposition
// TODO: mat_lu, mat_qr, mat_chol, mat_svd

// Eigenvalue
// TODO: mat_eig, mat_eigvals

#ifdef __cplusplus
}
#endif

#endif // MAT_H_

#ifdef MAT_IMPLEMENTATION

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

// Construction & Memory

MATDEF Mat *mat_empty(size_t rows, size_t cols) {
  MAT_ASSERT_DIM(rows, cols);

  Mat *mat = (Mat *)MAT_MALLOC(sizeof(Mat));
  mat->rows = rows;
  mat->cols = cols;
  mat->data = NULL;

  return mat;
}

MATDEF Mat *mat_mat(size_t rows, size_t cols) {
  Mat *mat = mat_empty(rows, cols);
  mat->data = (mat_elem_t *)MAT_CALLOC(rows * cols, sizeof(mat_elem_t));

  return mat;
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

MATDEF Vec *mat_vec(size_t dim) {
  Vec *vec = mat_mat(dim, 1);
  return vec;
}

MATDEF Vec *mat_vec_from(size_t dim, const mat_elem_t *values) {
  Vec *result = mat_from(dim, 1, values);

  return result;
}

MATDEF Mat *mat_copy(const Mat *m) {
  MAT_ASSERT_MAT(m);

  Mat *result = mat_mat(m->rows, m->cols);

  return result;
}

MATDEF Mat *mat_deep_copy(const Mat *m) {
  MAT_ASSERT_MAT(m);

  Mat *result = mat_copy(m);
  size_t len = m->rows * m->cols;

  memcpy(result->data, m->data, len * sizeof(mat_elem_t));

  return result;
}

// Accessors & Info

MATDEF mat_elem_t mat_at(const Mat *m, size_t row, size_t col) {
  MAT_ASSERT_MAT(m);
  return m->data[row * m->cols + col];
}

MATDEF MatSize mat_size(const Mat *m) {
  MatSize size = {m->rows, m->cols};
  return size;
}

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

// Comparison

MATDEF bool mat_equals(const Mat *a, const Mat *b) {
  return mat_equals_tol(a, b, MAT_DEFAULT_EPSILON);
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC bool mat_equals_tol_neon_impl(const Mat *a, const Mat *b, mat_elem_t epsilon) {
  size_t n = a->rows * a->cols;
  const mat_elem_t *pa = a->data;
  const mat_elem_t *pb = b->data;
  MAT_NEON_TYPE eps = MAT_NEON_DUP(epsilon);

  size_t i = 0;
  for (; i + MAT_NEON_WIDTH * 4 <= n; i += MAT_NEON_WIDTH * 4) {
    MAT_NEON_TYPE diff0 = MAT_NEON_ABD(MAT_NEON_LOAD(pa + i), MAT_NEON_LOAD(pb + i));
    MAT_NEON_TYPE diff1 = MAT_NEON_ABD(MAT_NEON_LOAD(pa + i + MAT_NEON_WIDTH), MAT_NEON_LOAD(pb + i + MAT_NEON_WIDTH));
    MAT_NEON_TYPE diff2 = MAT_NEON_ABD(MAT_NEON_LOAD(pa + i + MAT_NEON_WIDTH * 2), MAT_NEON_LOAD(pb + i + MAT_NEON_WIDTH * 2));
    MAT_NEON_TYPE diff3 = MAT_NEON_ABD(MAT_NEON_LOAD(pa + i + MAT_NEON_WIDTH * 3), MAT_NEON_LOAD(pb + i + MAT_NEON_WIDTH * 3));

    MAT_NEON_UTYPE gt0 = MAT_NEON_CGT(diff0, eps);
    MAT_NEON_UTYPE gt1 = MAT_NEON_CGT(diff1, eps);
    MAT_NEON_UTYPE gt2 = MAT_NEON_CGT(diff2, eps);
    MAT_NEON_UTYPE gt3 = MAT_NEON_CGT(diff3, eps);

    MAT_NEON_UTYPE any01 = MAT_NEON_ORR_U(gt0, gt1);
    MAT_NEON_UTYPE any23 = MAT_NEON_ORR_U(gt2, gt3);
    MAT_NEON_UTYPE any = MAT_NEON_ORR_U(any01, any23);

    if (MAT_NEON_MAXV_U(any) != 0) return false;
  }

  for (; i + MAT_NEON_WIDTH <= n; i += MAT_NEON_WIDTH) {
    MAT_NEON_TYPE diff = MAT_NEON_ABD(MAT_NEON_LOAD(pa + i), MAT_NEON_LOAD(pb + i));
    MAT_NEON_UTYPE gt = MAT_NEON_CGT(diff, eps);
    if (MAT_NEON_MAXV_U(gt) != 0) return false;
  }

  for (; i < n; i++) {
    mat_elem_t diff = pa[i] - pb[i];
    if (diff < 0) diff = -diff;
    if (diff > epsilon) return false;
  }

  return true;
}
#endif

MAT_INTERNAL_STATIC bool mat_equals_tol_scalar_impl(const Mat *a, const Mat *b, mat_elem_t epsilon) {
  size_t n = a->rows * a->cols;
  for (size_t i = 0; i < n; i++) {
    mat_elem_t diff = a->data[i] - b->data[i];
    if (diff < 0) diff = -diff;
    if (diff > epsilon) return false;
  }
  return true;
}

MATDEF bool mat_equals_tol(const Mat *a, const Mat *b, mat_elem_t epsilon) {
  MAT_ASSERT_MAT(a);
  MAT_ASSERT_MAT(b);

  if (a->rows != b->rows || a->cols != b->cols)
    return false;

#ifdef MAT_HAS_ARM_NEON
  return mat_equals_tol_neon_impl(a, b, epsilon);
#else
  return mat_equals_tol_scalar_impl(a, b, epsilon);
#endif
}

// Element-wise Unary

MATDEF void mat_abs(Mat *out, const Mat *a) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);

  size_t len = a->rows * a->cols;
  for (size_t i = 0; i < len; i++) out->data[i] = fabs(a->data[i]);
}

// Scalar Operations

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

// Matrix Arithmetic

MATDEF void mat_add(Mat *out, const Mat *a, const Mat *b) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);
  MAT_ASSERT_MAT(b);
  MAT_ASSERT(a->rows == b->rows);
  MAT_ASSERT(a->cols == b->cols);

  size_t len = a->rows * a->cols;
  for (size_t i = 0; i < len; i++) {
    out->data[i] = a->data[i] + b->data[i];
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

  size_t len = a->rows * a->cols;

  for (size_t i = 0; i < len; i++) {
    out->data[i] = a->data[i] - b->data[i];
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

// Matrix Products

MAT_INTERNAL_STATIC void mat_mul_scalar_impl(Mat *out, const Mat *a, const Mat *b) {
  size_t M = a->rows;
  size_t K = a->cols;
  size_t N = b->cols;

  // Zero output
  for (size_t i = 0; i < M * N; i++) {
    out->data[i] = 0;
  }

  // ikj loop order for cache-friendly access
  for (size_t i = 0; i < M; i++) {
    for (size_t k = 0; k < K; k++) {
      mat_elem_t aik = a->data[i * K + k];
      for (size_t j = 0; j < N; j++) {
        out->data[i * N + j] += aik * b->data[k * N + j];
      }
    }
  }
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void mat_mul_neon_impl(Mat *out, const Mat *a, const Mat *b) {
  size_t M = a->rows;
  size_t K = a->cols;
  size_t N = b->cols;

  // Transpose B for cache-friendly access
  mat_elem_t *bt = (mat_elem_t *)MAT_MALLOC(K * N * sizeof(mat_elem_t));
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < K; j++) {
      bt[i * K + j] = b->data[j * N + i];
    }
  }

  // 4x4 micro-kernel with 4x unrolling
  size_t i = 0;
  for (; i + 4 <= M; i += 4) {
    size_t j = 0;
    for (; j + 4 <= N; j += 4) {
      // 16 accumulators for 4x4 output tile
      MAT_NEON_TYPE acc00 = MAT_NEON_DUP(0), acc01 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc02 = MAT_NEON_DUP(0), acc03 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc10 = MAT_NEON_DUP(0), acc11 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc12 = MAT_NEON_DUP(0), acc13 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc20 = MAT_NEON_DUP(0), acc21 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc22 = MAT_NEON_DUP(0), acc23 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc30 = MAT_NEON_DUP(0), acc31 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc32 = MAT_NEON_DUP(0), acc33 = MAT_NEON_DUP(0);

      size_t k = 0;
      for (; k + MAT_NEON_WIDTH * 4 <= K; k += MAT_NEON_WIDTH * 4) {
        #define KERNEL_STEP(off) do { \
          MAT_NEON_TYPE a0 = MAT_NEON_LOAD(&a->data[(i+0)*K + k + (off)]); \
          MAT_NEON_TYPE a1 = MAT_NEON_LOAD(&a->data[(i+1)*K + k + (off)]); \
          MAT_NEON_TYPE a2 = MAT_NEON_LOAD(&a->data[(i+2)*K + k + (off)]); \
          MAT_NEON_TYPE a3 = MAT_NEON_LOAD(&a->data[(i+3)*K + k + (off)]); \
          MAT_NEON_TYPE b0 = MAT_NEON_LOAD(&bt[(j+0)*K + k + (off)]); \
          MAT_NEON_TYPE b1 = MAT_NEON_LOAD(&bt[(j+1)*K + k + (off)]); \
          MAT_NEON_TYPE b2 = MAT_NEON_LOAD(&bt[(j+2)*K + k + (off)]); \
          MAT_NEON_TYPE b3 = MAT_NEON_LOAD(&bt[(j+3)*K + k + (off)]); \
          acc00 = MAT_NEON_FMA(acc00, a0, b0); acc01 = MAT_NEON_FMA(acc01, a0, b1); \
          acc02 = MAT_NEON_FMA(acc02, a0, b2); acc03 = MAT_NEON_FMA(acc03, a0, b3); \
          acc10 = MAT_NEON_FMA(acc10, a1, b0); acc11 = MAT_NEON_FMA(acc11, a1, b1); \
          acc12 = MAT_NEON_FMA(acc12, a1, b2); acc13 = MAT_NEON_FMA(acc13, a1, b3); \
          acc20 = MAT_NEON_FMA(acc20, a2, b0); acc21 = MAT_NEON_FMA(acc21, a2, b1); \
          acc22 = MAT_NEON_FMA(acc22, a2, b2); acc23 = MAT_NEON_FMA(acc23, a2, b3); \
          acc30 = MAT_NEON_FMA(acc30, a3, b0); acc31 = MAT_NEON_FMA(acc31, a3, b1); \
          acc32 = MAT_NEON_FMA(acc32, a3, b2); acc33 = MAT_NEON_FMA(acc33, a3, b3); \
        } while(0)
        KERNEL_STEP(0);
        KERNEL_STEP(MAT_NEON_WIDTH);
        KERNEL_STEP(MAT_NEON_WIDTH * 2);
        KERNEL_STEP(MAT_NEON_WIDTH * 3);
        #undef KERNEL_STEP
      }

      for (; k + MAT_NEON_WIDTH <= K; k += MAT_NEON_WIDTH) {
        MAT_NEON_TYPE a0 = MAT_NEON_LOAD(&a->data[(i+0)*K + k]);
        MAT_NEON_TYPE a1 = MAT_NEON_LOAD(&a->data[(i+1)*K + k]);
        MAT_NEON_TYPE a2 = MAT_NEON_LOAD(&a->data[(i+2)*K + k]);
        MAT_NEON_TYPE a3 = MAT_NEON_LOAD(&a->data[(i+3)*K + k]);
        MAT_NEON_TYPE b0 = MAT_NEON_LOAD(&bt[(j+0)*K + k]);
        MAT_NEON_TYPE b1 = MAT_NEON_LOAD(&bt[(j+1)*K + k]);
        MAT_NEON_TYPE b2 = MAT_NEON_LOAD(&bt[(j+2)*K + k]);
        MAT_NEON_TYPE b3 = MAT_NEON_LOAD(&bt[(j+3)*K + k]);
        acc00 = MAT_NEON_FMA(acc00, a0, b0); acc01 = MAT_NEON_FMA(acc01, a0, b1);
        acc02 = MAT_NEON_FMA(acc02, a0, b2); acc03 = MAT_NEON_FMA(acc03, a0, b3);
        acc10 = MAT_NEON_FMA(acc10, a1, b0); acc11 = MAT_NEON_FMA(acc11, a1, b1);
        acc12 = MAT_NEON_FMA(acc12, a1, b2); acc13 = MAT_NEON_FMA(acc13, a1, b3);
        acc20 = MAT_NEON_FMA(acc20, a2, b0); acc21 = MAT_NEON_FMA(acc21, a2, b1);
        acc22 = MAT_NEON_FMA(acc22, a2, b2); acc23 = MAT_NEON_FMA(acc23, a2, b3);
        acc30 = MAT_NEON_FMA(acc30, a3, b0); acc31 = MAT_NEON_FMA(acc31, a3, b1);
        acc32 = MAT_NEON_FMA(acc32, a3, b2); acc33 = MAT_NEON_FMA(acc33, a3, b3);
      }

      // Horizontal sum and store
      out->data[(i+0)*N+(j+0)] = MAT_NEON_ADDV(acc00);
      out->data[(i+0)*N+(j+1)] = MAT_NEON_ADDV(acc01);
      out->data[(i+0)*N+(j+2)] = MAT_NEON_ADDV(acc02);
      out->data[(i+0)*N+(j+3)] = MAT_NEON_ADDV(acc03);
      out->data[(i+1)*N+(j+0)] = MAT_NEON_ADDV(acc10);
      out->data[(i+1)*N+(j+1)] = MAT_NEON_ADDV(acc11);
      out->data[(i+1)*N+(j+2)] = MAT_NEON_ADDV(acc12);
      out->data[(i+1)*N+(j+3)] = MAT_NEON_ADDV(acc13);
      out->data[(i+2)*N+(j+0)] = MAT_NEON_ADDV(acc20);
      out->data[(i+2)*N+(j+1)] = MAT_NEON_ADDV(acc21);
      out->data[(i+2)*N+(j+2)] = MAT_NEON_ADDV(acc22);
      out->data[(i+2)*N+(j+3)] = MAT_NEON_ADDV(acc23);
      out->data[(i+3)*N+(j+0)] = MAT_NEON_ADDV(acc30);
      out->data[(i+3)*N+(j+1)] = MAT_NEON_ADDV(acc31);
      out->data[(i+3)*N+(j+2)] = MAT_NEON_ADDV(acc32);
      out->data[(i+3)*N+(j+3)] = MAT_NEON_ADDV(acc33);

      // Scalar remainder k
      for (; k < K; k++) {
        mat_elem_t a0s = a->data[(i+0)*K + k], a1s = a->data[(i+1)*K + k];
        mat_elem_t a2s = a->data[(i+2)*K + k], a3s = a->data[(i+3)*K + k];
        for (size_t jj = 0; jj < 4; jj++) {
          mat_elem_t bval = bt[(j+jj)*K + k];
          out->data[(i+0)*N+(j+jj)] += a0s * bval;
          out->data[(i+1)*N+(j+jj)] += a1s * bval;
          out->data[(i+2)*N+(j+jj)] += a2s * bval;
          out->data[(i+3)*N+(j+jj)] += a3s * bval;
        }
      }
    }

    // Remainder j columns
    for (; j < N; j++) {
      for (size_t ii = 0; ii < 4; ii++) {
        mat_elem_t sum = 0;
        for (size_t k = 0; k < K; k++) {
          sum += a->data[(i+ii)*K + k] * bt[j*K + k];
        }
        out->data[(i+ii)*N + j] = sum;
      }
    }
  }

  // Remainder i rows
  for (; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      mat_elem_t sum = 0;
      for (size_t k = 0; k < K; k++) {
        sum += a->data[i*K + k] * bt[j*K + k];
      }
      out->data[i*N + j] = sum;
    }
  }

  MAT_FREE(bt);
}
#endif

MATDEF void mat_mul(Mat *out, const Mat *a, const Mat *b) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);
  MAT_ASSERT_MAT(b);
  MAT_ASSERT(a->cols == b->rows);

#ifdef MAT_HAS_ARM_NEON
  mat_mul_neon_impl(out, a, b);
#else
  mat_mul_scalar_impl(out, a, b);
#endif
}

MATDEF Mat *mat_rmul(const Mat *a, const Mat *b) {
  MAT_ASSERT_MAT(a);
  MAT_ASSERT_MAT(b);
  MAT_ASSERT(a->cols == b->rows);

  Mat *result = mat_mat(a->rows, b->cols);

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

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC mat_elem_t mat_dot_neon_impl(const Vec *v1, const Vec *v2) {
  size_t len = v1->rows;
  mat_elem_t *pa = v1->data;
  mat_elem_t *pb = v2->data;

  MAT_NEON_TYPE vsum0 = MAT_NEON_DUP(0);
  MAT_NEON_TYPE vsum1 = MAT_NEON_DUP(0);
  MAT_NEON_TYPE vsum2 = MAT_NEON_DUP(0);
  MAT_NEON_TYPE vsum3 = MAT_NEON_DUP(0);

  size_t i = 0;

  for (; i + MAT_NEON_WIDTH * 4 <= len; i += MAT_NEON_WIDTH * 4) {
    MAT_NEON_TYPE va0 = MAT_NEON_LOAD(&pa[i]);
    MAT_NEON_TYPE va1 = MAT_NEON_LOAD(&pa[i + MAT_NEON_WIDTH]);
    MAT_NEON_TYPE va2 = MAT_NEON_LOAD(&pa[i + MAT_NEON_WIDTH * 2]);
    MAT_NEON_TYPE va3 = MAT_NEON_LOAD(&pa[i + MAT_NEON_WIDTH * 3]);

    MAT_NEON_TYPE vb0 = MAT_NEON_LOAD(&pb[i]);
    MAT_NEON_TYPE vb1 = MAT_NEON_LOAD(&pb[i + MAT_NEON_WIDTH]);
    MAT_NEON_TYPE vb2 = MAT_NEON_LOAD(&pb[i + MAT_NEON_WIDTH * 2]);
    MAT_NEON_TYPE vb3 = MAT_NEON_LOAD(&pb[i + MAT_NEON_WIDTH * 3]);

    vsum0 = MAT_NEON_FMA(vsum0, va0, vb0);
    vsum1 = MAT_NEON_FMA(vsum1, va1, vb1);
    vsum2 = MAT_NEON_FMA(vsum2, va2, vb2);
    vsum3 = MAT_NEON_FMA(vsum3, va3, vb3);
  }

  for (; i + MAT_NEON_WIDTH <= len; i += MAT_NEON_WIDTH) {
    MAT_NEON_TYPE va = MAT_NEON_LOAD(&pa[i]);
    MAT_NEON_TYPE vb = MAT_NEON_LOAD(&pb[i]);
    vsum0 = MAT_NEON_FMA(vsum0, va, vb);
  }

  vsum0 = MAT_NEON_ADD(vsum0, vsum1);
  vsum2 = MAT_NEON_ADD(vsum2, vsum3);
  vsum0 = MAT_NEON_ADD(vsum0, vsum2);
  mat_elem_t result = MAT_NEON_ADDV(vsum0);

  for (; i < len; i++) {
    result += pa[i] * pb[i];
  }

  return result;
}
#endif

MAT_INTERNAL_STATIC mat_elem_t mat_dot_scalar_impl(const Vec *v1, const Vec *v2) {
  mat_elem_t result = 0;
  for (size_t i = 0; i < v1->rows; i++) {
    result += v1->data[i] * v2->data[i];
  }
  return result;
}

MATDEF mat_elem_t mat_dot(const Vec *v1, const Vec *v2) {
  MAT_ASSERT_MAT(v1);
  MAT_ASSERT_MAT(v2);

#ifdef MAT_HAS_ARM_NEON
  return mat_dot_neon_impl(v1, v2);
#else
  return mat_dot_scalar_impl(v1, v2);
#endif
}

// Structure Operations

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

MATDEF Mat *mat_rt(const Mat *m) {
  MAT_ASSERT_MAT(m);

  Mat *result = mat_mat(m->cols, m->rows);
  mat_t(result, m);

  return result;
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

// Diagonal Operations

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

// Norms

MATDEF mat_elem_t mat_norm(const Mat *a, size_t p) {
  MAT_ASSERT_MAT(a);
  MAT_ASSERT(p >= 1);

  size_t len = a->rows * a->cols;
  mat_elem_t sum = 0;

  for (size_t i = 0; i < len; i++) {
    sum += pow(fabs(a->data[i]), p);
  }

  return pow(sum, 1.0 / p);
}

MATDEF mat_elem_t mat_norm2(const Mat *a) {
  return mat_norm(a, 2);
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC mat_elem_t mat_norm_max_neon_impl(const Mat *a) {
  size_t len = a->rows * a->cols;
  mat_elem_t *pa = a->data;

  MAT_NEON_TYPE vmax0 = MAT_NEON_DUP(0);
  MAT_NEON_TYPE vmax1 = MAT_NEON_DUP(0);
  MAT_NEON_TYPE vmax2 = MAT_NEON_DUP(0);
  MAT_NEON_TYPE vmax3 = MAT_NEON_DUP(0);

  size_t i = 0;
  for (; i + MAT_NEON_WIDTH * 4 <= len; i += MAT_NEON_WIDTH * 4) {
    MAT_NEON_TYPE va0 = MAT_NEON_ABS(MAT_NEON_LOAD(&pa[i]));
    MAT_NEON_TYPE va1 = MAT_NEON_ABS(MAT_NEON_LOAD(&pa[i + MAT_NEON_WIDTH]));
    MAT_NEON_TYPE va2 = MAT_NEON_ABS(MAT_NEON_LOAD(&pa[i + MAT_NEON_WIDTH * 2]));
    MAT_NEON_TYPE va3 = MAT_NEON_ABS(MAT_NEON_LOAD(&pa[i + MAT_NEON_WIDTH * 3]));

    vmax0 = MAT_NEON_MAX(vmax0, va0);
    vmax1 = MAT_NEON_MAX(vmax1, va1);
    vmax2 = MAT_NEON_MAX(vmax2, va2);
    vmax3 = MAT_NEON_MAX(vmax3, va3);
  }

  for (; i + MAT_NEON_WIDTH <= len; i += MAT_NEON_WIDTH) {
    MAT_NEON_TYPE va = MAT_NEON_ABS(MAT_NEON_LOAD(&pa[i]));
    vmax0 = MAT_NEON_MAX(vmax0, va);
  }

  vmax0 = MAT_NEON_MAX(vmax0, vmax1);
  vmax2 = MAT_NEON_MAX(vmax2, vmax3);
  vmax0 = MAT_NEON_MAX(vmax0, vmax2);
  mat_elem_t max = MAT_NEON_MAXV(vmax0);

  for (; i < len; i++) {
    mat_elem_t v = fabs(pa[i]);
    if (v > max) max = v;
  }

  return max;
}
#endif

MAT_INTERNAL_STATIC mat_elem_t mat_norm_max_scalar_impl(const Mat *a) {
  size_t len = a->rows * a->cols;
  mat_elem_t max = fabs(a->data[0]);
  for (size_t i = 1; i < len; i++) {
    mat_elem_t v = fabs(a->data[i]);
    if (v > max) max = v;
  }
  return max;
}

MATDEF mat_elem_t mat_norm_max(const Mat *a) {
  MAT_ASSERT_MAT(a);

#ifdef MAT_HAS_ARM_NEON
  return mat_norm_max_neon_impl(a);
#else
  return mat_norm_max_scalar_impl(a);
#endif
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC mat_elem_t mat_norm_fro_neon_impl(const Mat *a) {
  size_t len = a->rows * a->cols;
  mat_elem_t *pa = a->data;

  MAT_NEON_TYPE vsum0 = MAT_NEON_DUP(0);
  MAT_NEON_TYPE vsum1 = MAT_NEON_DUP(0);
  MAT_NEON_TYPE vsum2 = MAT_NEON_DUP(0);
  MAT_NEON_TYPE vsum3 = MAT_NEON_DUP(0);

  size_t i = 0;
  for (; i + MAT_NEON_WIDTH * 4 <= len; i += MAT_NEON_WIDTH * 4) {
    MAT_NEON_TYPE va0 = MAT_NEON_LOAD(&pa[i]);
    MAT_NEON_TYPE va1 = MAT_NEON_LOAD(&pa[i + MAT_NEON_WIDTH]);
    MAT_NEON_TYPE va2 = MAT_NEON_LOAD(&pa[i + MAT_NEON_WIDTH * 2]);
    MAT_NEON_TYPE va3 = MAT_NEON_LOAD(&pa[i + MAT_NEON_WIDTH * 3]);

    vsum0 = MAT_NEON_FMA(vsum0, va0, va0);
    vsum1 = MAT_NEON_FMA(vsum1, va1, va1);
    vsum2 = MAT_NEON_FMA(vsum2, va2, va2);
    vsum3 = MAT_NEON_FMA(vsum3, va3, va3);
  }

  for (; i + MAT_NEON_WIDTH <= len; i += MAT_NEON_WIDTH) {
    MAT_NEON_TYPE va = MAT_NEON_LOAD(&pa[i]);
    vsum0 = MAT_NEON_FMA(vsum0, va, va);
  }

  vsum0 = MAT_NEON_ADD(vsum0, vsum1);
  vsum2 = MAT_NEON_ADD(vsum2, vsum3);
  vsum0 = MAT_NEON_ADD(vsum0, vsum2);
  mat_elem_t sum = MAT_NEON_ADDV(vsum0);

  for (; i < len; i++) {
    sum += pa[i] * pa[i];
  }

  return sqrt(sum);
}
#endif

MAT_INTERNAL_STATIC mat_elem_t mat_norm_fro_scalar_impl(const Mat *a) {
  size_t len = a->rows * a->cols;
  mat_elem_t sum = 0;
  for (size_t i = 0; i < len; i++) {
    sum += a->data[i] * a->data[i];
  }
  return sqrt(sum);
}

MATDEF mat_elem_t mat_norm_fro(const Mat *a) {
  MAT_ASSERT_MAT(a);

#ifdef MAT_HAS_ARM_NEON
  return mat_norm_fro_neon_impl(a);
#else
  return mat_norm_fro_scalar_impl(a);
#endif
}

// Matrix Properties

MATDEF mat_elem_t mat_trace(const Mat *a) {
  MAT_ASSERT_MAT(a);
  MAT_ASSERT_SQUARE(a);

  size_t dim = a->rows;
  mat_elem_t *pa = a->data;
  size_t stride = dim + 1;

  // Multiple accumulators to break dependency chain
  mat_elem_t sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;

  size_t i = 0;
  for (; i + 4 <= dim; i += 4) {
    sum0 += pa[i * stride];
    sum1 += pa[(i + 1) * stride];
    sum2 += pa[(i + 2) * stride];
    sum3 += pa[(i + 3) * stride];
  }

  mat_elem_t result = sum0 + sum1 + sum2 + sum3;

  for (; i < dim; i++) {
    result += pa[i * stride];
  }

  return result;
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC mat_elem_t mat_nnz_neon_impl(const Mat *a) {
  size_t len = a->rows * a->cols;
  mat_elem_t *pa = a->data;

  MAT_NEON_TYPE vzero = MAT_NEON_DUP(0);
  MAT_NEON_UTYPE vcount0 = MAT_NEON_DUP_U(0);
  MAT_NEON_UTYPE vcount1 = MAT_NEON_DUP_U(0);
  MAT_NEON_UTYPE vcount2 = MAT_NEON_DUP_U(0);
  MAT_NEON_UTYPE vcount3 = MAT_NEON_DUP_U(0);
  MAT_NEON_UTYPE vone = MAT_NEON_DUP_U(1);

  size_t i = 0;
  for (; i + MAT_NEON_WIDTH * 4 <= len; i += MAT_NEON_WIDTH * 4) {
    MAT_NEON_TYPE va0 = MAT_NEON_LOAD(&pa[i]);
    MAT_NEON_TYPE va1 = MAT_NEON_LOAD(&pa[i + MAT_NEON_WIDTH]);
    MAT_NEON_TYPE va2 = MAT_NEON_LOAD(&pa[i + MAT_NEON_WIDTH * 2]);
    MAT_NEON_TYPE va3 = MAT_NEON_LOAD(&pa[i + MAT_NEON_WIDTH * 3]);

    MAT_NEON_UTYPE mask0 = MAT_NEON_AND_U(MAT_NEON_MVN_U(MAT_NEON_CEQ(va0, vzero)), vone);
    MAT_NEON_UTYPE mask1 = MAT_NEON_AND_U(MAT_NEON_MVN_U(MAT_NEON_CEQ(va1, vzero)), vone);
    MAT_NEON_UTYPE mask2 = MAT_NEON_AND_U(MAT_NEON_MVN_U(MAT_NEON_CEQ(va2, vzero)), vone);
    MAT_NEON_UTYPE mask3 = MAT_NEON_AND_U(MAT_NEON_MVN_U(MAT_NEON_CEQ(va3, vzero)), vone);

    vcount0 = MAT_NEON_ADD_U(vcount0, mask0);
    vcount1 = MAT_NEON_ADD_U(vcount1, mask1);
    vcount2 = MAT_NEON_ADD_U(vcount2, mask2);
    vcount3 = MAT_NEON_ADD_U(vcount3, mask3);
  }

  vcount0 = MAT_NEON_ADD_U(vcount0, vcount1);
  vcount2 = MAT_NEON_ADD_U(vcount2, vcount3);
  vcount0 = MAT_NEON_ADD_U(vcount0, vcount2);
  size_t count = MAT_NEON_ADDV_U(vcount0);

  for (; i < len; i++) {
    if (pa[i] != 0) count++;
  }

  return (mat_elem_t)count;
}
#endif

MAT_INTERNAL_STATIC mat_elem_t mat_nnz_scalar_impl(const Mat *a) {
  size_t len = a->rows * a->cols;
  mat_elem_t count = 0;
  for (size_t i = 0; i < len; i++) {
    if (a->data[i] != 0) count++;
  }
  return count;
}

MATDEF mat_elem_t mat_nnz(const Mat *a) {
  MAT_ASSERT_MAT(a);

#ifdef MAT_HAS_ARM_NEON
  return mat_nnz_neon_impl(a);
#else
  return mat_nnz_scalar_impl(a);
#endif
}

#endif // MAT_IMPLEMENTATION
