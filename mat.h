#ifndef MAT_H_
#define MAT_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Column-major storage is always enabled (required for optimal BLAS performance)
#define MAT_COLUMN_MAJOR

// ARM NEON detection
#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
#define MAT_HAS_ARM_NEON
#include <arm_neon.h>
#endif

// OpenMP detection (auto-enabled when compiled with -fopenmp)
#if defined(_OPENMP)
#include <omp.h>
#define MAT_HAS_OPENMP
#ifndef MAT_OMP_THRESHOLD
#define MAT_OMP_THRESHOLD                                                      \
  (1024 * 1024) // Skip parallelization for small matrices
#endif
#endif

#ifndef MATDEF
#define MATDEF
#endif

// Mark experimental functions (warns on use)
#if defined(__GNUC__)
#define MAT_EXPERIMENTAL                                                       \
  __attribute__((warning("experimental: does not guarantee correct results")))
#elif defined(_MSC_VER)
#define MAT_EXPERIMENTAL                                                       \
  __declspec(deprecated("experimental: does not guarantee correct results"))
#else
#define MAT_EXPERIMENTAL
#endif

// Mark unimplemented functions (errors on use)
#if defined(__GNUC__)
#define MAT_NOT_IMPLEMENTED __attribute__((error("not implemented")))
#elif defined(_MSC_VER)
#define MAT_NOT_IMPLEMENTED                                                    \
  __declspec(deprecated("not implemented - will not link"))
#else
#define MAT_NOT_IMPLEMENTED
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

#ifndef MAT_FREE_MAT
#define MAT_FREE_MAT(m)                                                        \
  do {                                                                         \
    MAT_FREE((m)->data);                                                       \
    MAT_FREE(m);                                                               \
  } while (0)
#endif

#ifndef MAT_FREE_PERM
#define MAT_FREE_PERM(p)                                                       \
  do {                                                                         \
    MAT_FREE((p)->data);                                                       \
    MAT_FREE(p);                                                               \
  } while (0)
#endif

// Scratch arena for temporary allocations in hot paths
// Define MAT_NO_SCRATCH to disable (uses malloc/free instead)
// Define MAT_SCRATCH_SIZE to override the default size
#ifndef MAT_SCRATCH_SIZE
#define MAT_SCRATCH_SIZE (4 * 1024 * 1024) // 4MB default
#endif

#ifndef MAT_NO_SCRATCH
typedef struct {
  char *buf;
  size_t offset;
  size_t size;
} MatArena;
#endif

// Transpose flags for GEMM operations (BLAS-style)
typedef enum {
  MAT_NO_TRANS = 0,  // Use matrix as-is
  MAT_TRANS = 1      // Use transpose of matrix
} mat_trans_t;

#ifdef MAT_STRIP_PREFIX

#define mat mat_mat
#define empty mat_empty
#define zeros mat_zeros
#define ones mat_ones
#define eye mat_reye
#define reye mat_reye
#define deep_copy mat_deep_copy
#define rdeep_copy mat_rdeep_copy
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
#define mat_new(cols, ...)                                                     \
  mat_from(sizeof((mat_elem_t[][cols])__VA_ARGS__) / sizeof(mat_elem_t[cols]), \
           cols, (mat_elem_t *)((mat_elem_t[][cols])__VA_ARGS__))

#define mat_set(out, ...) mat_init(out, (mat_elem_t[])__VA_ARGS__)

#define mat_vnew(...)                                                          \
  mat_vec_from(sizeof((mat_elem_t[])__VA_ARGS__) / sizeof(mat_elem_t),         \
               (mat_elem_t[])__VA_ARGS__)

#define mat_rnew(...)                                                          \
  mat_from(1, sizeof((mat_elem_t[])__VA_ARGS__) / sizeof(mat_elem_t),          \
           (mat_elem_t[])__VA_ARGS__)

// Element type (float or double precision)
#ifdef MAT_DOUBLE_PRECISION
typedef double mat_elem_t;
#ifndef MAT_DEFAULT_EPSILON
#define MAT_DEFAULT_EPSILON 1e-9
#endif
#define MAT_FABS fabs
#define MAT_SQRT sqrt
#define MAT_HUGE_VAL HUGE_VAL
#else
typedef float mat_elem_t;
#ifndef MAT_DEFAULT_EPSILON
#define MAT_DEFAULT_EPSILON 1e-6f
#endif
#define MAT_FABS fabsf
#define MAT_SQRT sqrtf
#define MAT_HUGE_VAL HUGE_VALF
#endif

// NEON SIMD macros (only defined when targeting ARM with NEON)
#ifdef MAT_HAS_ARM_NEON
#ifdef MAT_DOUBLE_PRECISION
// NEON double precision (2 doubles per 128-bit register)
#define MAT_NEON_TYPE float64x2_t
#define MAT_NEON_UTYPE uint64x2_t
#define MAT_NEON_WIDTH 2
#define MAT_NEON_LOAD vld1q_f64
#define MAT_NEON_STORE vst1q_f64
#define MAT_NEON_DUP vdupq_n_f64
#define MAT_NEON_DUP_U vdupq_n_u64
#define MAT_NEON_FMA vfmaq_f64
#define MAT_NEON_FMA_LANE vfmaq_laneq_f64
#define MAT_NEON_FMS vfmsq_f64
#define MAT_NEON_ADD vaddq_f64
#define MAT_NEON_ADDV vaddvq_f64
#define MAT_NEON_ABS vabsq_f64
#define MAT_NEON_MAX vmaxq_f64
#define MAT_NEON_MAXV vmaxvq_f64
#define MAT_NEON_ABD vabdq_f64
#define MAT_NEON_CGT vcgtq_f64
#define MAT_NEON_CEQ vceqq_f64
#define MAT_NEON_ORR_U vorrq_u64
#define MAT_NEON_AND_U vandq_u64
#define MAT_NEON_MVN_U(x) veorq_u64(x, vdupq_n_u64(0xFFFFFFFFFFFFFFFFULL))
#define MAT_NEON_MAXV_U(x) (vgetq_lane_u64(x, 0) | vgetq_lane_u64(x, 1))
#define MAT_NEON_ADDV_U(x) (vgetq_lane_u64(x, 0) + vgetq_lane_u64(x, 1))
#define MAT_NEON_ADD_U vaddq_u64
#define MAT_NEON_MUL vmulq_f64
#define MAT_NEON_SUB vsubq_f64
#define MAT_NEON_ZIP1 vzip1q_f64
#define MAT_NEON_ZIP2 vzip2q_f64
#define MAT_NEON_GET_LANE(v, n) vgetq_lane_f64(v, n)
#else
// NEON single precision (4 floats per 128-bit register)
#define MAT_NEON_TYPE float32x4_t
#define MAT_NEON_UTYPE uint32x4_t
#define MAT_NEON_WIDTH 4
#define MAT_NEON_LOAD vld1q_f32
#define MAT_NEON_STORE vst1q_f32
#define MAT_NEON_DUP vdupq_n_f32
#define MAT_NEON_DUP_U vdupq_n_u32
#define MAT_NEON_FMA vfmaq_f32
#define MAT_NEON_FMA_LANE vfmaq_laneq_f32
#define MAT_NEON_FMS vfmsq_f32
#define MAT_NEON_ADD vaddq_f32
#define MAT_NEON_ADDV vaddvq_f32
#define MAT_NEON_ABS vabsq_f32
#define MAT_NEON_MAX vmaxq_f32
#define MAT_NEON_MAXV vmaxvq_f32
#define MAT_NEON_ABD vabdq_f32
#define MAT_NEON_CGT vcgtq_f32
#define MAT_NEON_CEQ vceqq_f32
#define MAT_NEON_ORR_U vorrq_u32
#define MAT_NEON_AND_U vandq_u32
#define MAT_NEON_MVN_U(x) vmvnq_u32(x)
#define MAT_NEON_MAXV_U(x) vmaxvq_u32(x)
#define MAT_NEON_ADDV_U(x) vaddvq_u32(x)
#define MAT_NEON_ADD_U vaddq_u32
#define MAT_NEON_MUL vmulq_f32
#define MAT_NEON_SUB vsubq_f32
#define MAT_NEON_ZIP1 vzip1q_f32
#define MAT_NEON_ZIP2 vzip2q_f32
#define MAT_NEON_GET_LANE(v, n) vgetq_lane_f32(v, n)
#endif

// Accumulator macros for overflow-safe reductions (always use double)
#define MAT_ACC_TYPE float64x2_t
#define MAT_ACC_ZERO vdupq_n_f64(0)
#define MAT_ACC_ADD vaddq_f64
#define MAT_ACC_ADDV vaddvq_f64
#define MAT_ACC_FMA vfmaq_f64
#ifdef MAT_DOUBLE_PRECISION
#define MAT_ACC_WIDTH 2
#define MAT_ACC_LOAD_SQ(acc, ptr)                                              \
  do {                                                                         \
    float64x2_t _v = vld1q_f64(ptr);                                           \
    acc = vfmaq_f64(acc, _v, _v);                                              \
  } while (0)
#else
#define MAT_ACC_WIDTH 2
#define MAT_ACC_LOAD_SQ(acc, ptr)                                              \
  do {                                                                         \
    float32x2_t _v = vld1_f32(ptr);                                            \
    float64x2_t _d = vcvt_f64_f32(_v);                                         \
    acc = vfmaq_f64(acc, _d, _d);                                              \
  } while (0)
#endif
#endif // __ARM_NEON

// Control visibility of internal implementations
#ifdef MAT_EXPOSE_INTERNALS
#define MAT_INTERNAL_STATIC
#else
#ifdef __GNUC__
#define MAT_INTERNAL_STATIC static __attribute__((unused))
#else
#define MAT_INTERNAL_STATIC static
#endif
#endif

#define identity reye

#ifndef MAT_LOG_LEVEL
#define MAT_LOG_LEVEL 0
#endif

// The library provides logging functions as an utility.
// This is generally used for development of the library,
// but can completely be used by any user
#if MAT_LOG_LEVEL > 0
#include <stdio.h>
#ifndef MAT_LOG_OUTPUT_ERR
#define MAT_LOG_OUTPUT_ERR(msg) fprintf(stderr, "%s\n", msg)
#endif
#ifndef MAT_LOG_OUTPUT
#define MAT_LOG_OUTPUT(msg) fprintf(stdout, "%s\n", msg)
#endif
#endif

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

// It is possible to disable the assertions in the library if
// the user wants to. In general, the assert statements add a bit
// of overhead but most of the scenarios it is neglectable
#ifndef MAT_ASSERT
#include <assert.h>
#define MAT_ASSERT(x) assert(x)
#endif

// Utility assertion macros for specific types in the library.
// Used across all the library implementations but available
// for any user as well
#ifndef MAT_ASSERT_MAT
#define MAT_ASSERT_MAT(m)                                                      \
  do {                                                                         \
    MAT_ASSERT((m) != NULL);                                                   \
    MAT_ASSERT((m)->data != NULL);                                             \
    MAT_ASSERT((m)->rows > 0 && (m)->cols > 0);                                \
  } while (0)
#endif

#ifndef MAT_ASSERT_DIM
#define MAT_ASSERT_DIM(rows, cols)                                             \
  do {                                                                         \
    MAT_ASSERT((rows) > 0);                                                    \
    MAT_ASSERT((cols) > 0);                                                    \
  } while (0)
#endif

#ifndef MAT_ASSERT_SQUARE
#define MAT_ASSERT_SQUARE(m)                                                   \
  do {                                                                         \
    MAT_ASSERT_MAT(m);                                                         \
    MAT_ASSERT((m)->rows == (m)->cols);                                        \
  } while (0)
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Core matrix type
typedef struct {
  size_t rows;
  size_t cols;
  mat_elem_t *data;
} Mat;

// An utility type to provide the size of a given matrix
typedef struct {
  size_t x;
  size_t y;
} MatSize;

// Vectors in this library are implemented as a Mat
// of size nx1 (column vector) by default.
// Row vectors are just transposed column vectors.
typedef Mat Vec;

// Permutation type for better handling of such data containers.
// Permutations are vectors of integers, thus the need to have a
// separate type besides Vec.
typedef struct {
  size_t *data;
  size_t size;
} Perm;

// Storage order: Column-major (Fortran/BLAS style, column-contiguous)
// data[j * rows + i] accesses element at row i, column j
#define MAT_IDX(m, i, j) ((j) * (m)->rows + (i))

// Element access macros - use these instead of direct indexing
#define MAT_AT(m, i, j) ((m)->data[MAT_IDX(m, i, j)])
#define MAT_SET(m, i, j, v) ((m)->data[MAT_IDX(m, i, j)] = (v))

/* Construction & Memory */

// Allocate matrix struct without data buffer. Use for custom memory management.
MATDEF Mat *mat_empty(size_t rows, size_t cols);

// Allocate zero-initialized matrix.
MATDEF Mat *mat_mat(size_t rows, size_t cols);

// Create matrix from array of values (row-major order).
MATDEF Mat *mat_from(size_t rows, size_t cols, const mat_elem_t *values);

// Initialize existing matrix with values (row-major order).
MATDEF void mat_init(Mat *out, const mat_elem_t *values);

// Free matrix and its data buffer.
MATDEF void mat_free_mat(Mat *m);

// Create matrix filled with zeros.
MATDEF Mat *mat_zeros(size_t rows, size_t cols);

// Create matrix filled with ones.
MATDEF Mat *mat_ones(size_t rows, size_t cols);

// Create a matrix filled with any given number
MATDEF void mat_fill(Mat *out, mat_elem_t value);

// Set matrix to identity (must be square).
MATDEF void mat_eye(Mat *out);

// Create identity matrix of given dimension.
MATDEF Mat *mat_reye(size_t dim);

// Allocate zero-initialized column vector.
MATDEF Vec *mat_vec(size_t dim);

// Allocate a zero-initialized row vector.
MATDEF Vec *mat_row_vec(size_t dim);

// Create column vector from array of values.
MATDEF Vec *mat_vec_from(size_t dim, const mat_elem_t *values);

// Allocate permutation of given size.
MATDEF Perm *mat_perm(size_t n);

// Free permutation and its data.
MATDEF void mat_free_perm(Perm *p);

// Set permutation to identity [0, 1, 2, ...].
MATDEF void mat_perm_identity(Perm *p);

// Convert permutation to explicit n×n permutation matrix.
MATDEF Mat *mat_perm_mat(const Perm *p);

// Shallow copy (copies struct, shares data pointer). Use mat_rdeep_copy for
// full copy.
MATDEF Mat *mat_copy(const Mat *m);

// Deep copy src into pre-allocated out.
MATDEF void mat_deep_copy(Mat *out, const Mat *src);

// Allocate and return deep copy.
MATDEF Mat *mat_rdeep_copy(const Mat *m);

/* Accessors & Info */

// Get element at (row, col). Bounds checked via MAT_ASSERT.
MATDEF mat_elem_t mat_at(const Mat *mat, size_t row, size_t col);

// Set element at (row, col). Bounds checked via MAT_ASSERT.
MATDEF void mat_set_at(Mat *m, size_t row, size_t col, mat_elem_t value);

// Get matrix dimensions as {rows, cols} struct.
MATDEF MatSize mat_size(const Mat *m);

// Print matrix to stdout in MATLAB-like format.
MATDEF void mat_print(const Mat *m);

/* Comparison */

// Exact equality. Returns true if all elements are bitwise equal.
MATDEF bool mat_equals(const Mat *a, const Mat *b);

// Approximate equality. Returns true if max|a-b| < epsilon.
MATDEF bool mat_equals_tol(const Mat *a, const Mat *b, mat_elem_t epsilon);

/* Element-wise Unary */

// NOTE: Transcendental functions (exp, log, sin, cos, atan2) use scalar libc
// calls. For SIMD-optimized versions, consider libraries like SLEEF or Eigen.

// out[i] = |a[i]|
MATDEF void mat_abs(Mat *out, const Mat *a);

// out[i] = sqrt(a[i]). Undefined for negative inputs.
MATDEF void mat_sqrt(Mat *out, const Mat *a);

// out[i] = e^a[i]
MATDEF void mat_exp(Mat *out, const Mat *a);

// out[i] = ln(a[i]). Undefined for non-positive inputs.
MATDEF void mat_log(Mat *out, const Mat *a);

// out[i] = log10(a[i]). Undefined for non-positive inputs.
MATDEF void mat_log10(Mat *out, const Mat *a);

// out[i] = sin(a[i]), a in radians.
MATDEF void mat_sin(Mat *out, const Mat *a);

// out[i] = cos(a[i]), a in radians.
MATDEF void mat_cos(Mat *out, const Mat *a);

// out[i] = a[i]^exp. Undefined for negative base with fractional exponent.
MATDEF void mat_pow(Mat *out, const Mat *a, mat_elem_t exp);

// out[i] = clamp(a[i], min_val, max_val).
MATDEF void mat_clip(Mat *out, const Mat *a, mat_elem_t min_val,
                     mat_elem_t max_val);

// Element-wise Binary

// out[i] = a[i] / b[i]. No division-by-zero check.
MATDEF void mat_div(Mat *out, const Mat *a, const Mat *b);

// out[i] = atan2(y[i], x[i]). Result in radians, range [-pi, pi].
MATDEF void mat_atan2(Mat *out, const Mat *y, const Mat *x);

/* Scalar Operations */

// out[i] *= k. In-place scaling. SIMD-optimized.
MATDEF void mat_scale(Mat *out, mat_elem_t k);

// Return m scaled by k. Allocates new matrix.
MATDEF Mat *mat_rscale(const Mat *m, mat_elem_t k);

// Normalize vector in place: v = v / ||v||. Returns the norm.
// If norm < MAT_DEFAULT_EPSILON, vector is unchanged and 0 is returned.
// SIMD-optimized.
MATDEF mat_elem_t mat_normalize(Mat *v);

// out[i] += k. In-place scalar addition.
MATDEF void mat_add_scalar(Mat *out, mat_elem_t k);

// Return m + k. Allocates new matrix.
MATDEF Mat *mat_radd_scalar(const Mat *m, mat_elem_t k);

// Matrix Arithmetic

// out = a + b. SIMD-optimized.
MATDEF void mat_add(Mat *out, const Mat *a, const Mat *b);

// Return a + b. Allocates new matrix.
MATDEF Mat *mat_radd(const Mat *a, const Mat *b);

// out = a - b. SIMD-optimized.
MATDEF void mat_sub(Mat *out, const Mat *a, const Mat *b);

// Return a - b. Allocates new matrix.
MATDEF Mat *mat_rsub(const Mat *a, const Mat *b);

// out = sum of count matrices (variadic). Modifies out in-place.
MATDEF void mat_add_many(Mat *out, size_t count, ...);

// Return sum of count matrices (variadic). Allocates new matrix.
MATDEF Mat *mat_radd_many(size_t count, ...);

// Matrix Products

// out = a * b (matrix multiplication). SIMD-optimized.
// Dimensions: a(m,k) * b(k,n) = out(m,n).
MATDEF void mat_mul(Mat *out, const Mat *a, const Mat *b);

// Return a * b. Allocates new matrix.
MATDEF Mat *mat_rmul(const Mat *a, const Mat *b);

// out[i] = a[i] * b[i] (element-wise/Hadamard product). SIMD-optimized.
MATDEF void mat_hadamard(Mat *out, const Mat *a, const Mat *b);

// Return element-wise a * b. Allocates new matrix.
MATDEF Mat *mat_rhadamard(const Mat *a, const Mat *b);

// Return sum(v1[i] * v2[i]) (dot/inner product). SIMD-optimized.
MATDEF mat_elem_t mat_dot(const Vec *v1, const Vec *v2);

// out = v1 x v2 (cross product). Vectors must be 3D.
MATDEF void mat_cross(Vec *out, const Vec *v1, const Vec *v2);

// out = v1 * v2^T (outer product). out(m,n) where v1 is m-dim, v2 is n-dim.
// SIMD-optimized.
MATDEF void mat_outer(Mat *out, const Vec *v1, const Vec *v2);

// Return x^T * A * y (bilinear form). x is m-dim, A is m x n, y is n-dim.
MATDEF mat_elem_t mat_bilinear(const Vec *x, const Mat *A, const Vec *y);

// Return x^T * A * x (quadratic form). x is n-dim, A is n x n.
MATDEF mat_elem_t mat_quadform(const Vec *x, const Mat *A);

// Fused Operations (BLAS-like)

// y = alpha * x + y (AXPY). SIMD-optimized.
MATDEF void mat_axpy(Vec *y, mat_elem_t alpha, const Vec *x);

// y = alpha * A * x + beta * y (GEMV). SIMD-optimized.
MATDEF void mat_gemv(Vec *y, mat_elem_t alpha, const Mat *A, const Vec *x,
                     mat_elem_t beta);

// y = alpha * A^T * x + beta * y (GEMV transposed). SIMD-optimized.
MATDEF void mat_gemv_t(Vec *y, mat_elem_t alpha, const Mat *A, const Vec *x,
                       mat_elem_t beta);

// A = A + alpha * x * y^T (GER/rank-1 update).
MATDEF void mat_ger(Mat *A, mat_elem_t alpha, const Vec *x, const Vec *y);

// A = alpha * x * x^T + A (SYR). SIMD-optimized.
// uplo: 'L' for lower, 'U' for upper triangle.
MATDEF void mat_syr(Mat *A, mat_elem_t alpha, const Vec *x, char uplo);

// C = alpha * A * B + beta * C (GEMM). SIMD-optimized.
MATDEF void mat_gemm(Mat *C, mat_elem_t alpha, const Mat *A, const Mat *B,
                     mat_elem_t beta);

// C = alpha * A * A^T + beta * C (SYRK). SIMD-optimized.
// uplo: 'L' for lower, 'U' for upper triangle.
MATDEF void mat_syrk(Mat *C, const Mat *A, mat_elem_t alpha, mat_elem_t beta,
                     char uplo);

// C = alpha * A^T * A + beta * C (SYRK transposed). SIMD-optimized.
// uplo: 'L' for lower, 'U' for upper triangle.
MATDEF void mat_syrk_t(Mat *C, const Mat *A, mat_elem_t alpha, mat_elem_t beta,
                       char uplo);

// Structure Operations

// out = m^T (transpose). out must be pre-allocated with swapped dimensions.
MATDEF void mat_t(Mat *out, const Mat *m);

// Return m^T. Allocates new matrix.
MATDEF Mat *mat_rt(const Mat *m);

// Reshape out in-place. Total elements must remain constant.
MATDEF void mat_reshape(Mat *out, size_t rows, size_t cols);

// Return reshaped copy of m. Allocates new matrix.
MATDEF Mat *mat_rreshape(const Mat *m, size_t rows, size_t cols);

// out = [a, b] (horizontal concatenation). a and b must have same row count.
MATDEF void mat_hcat(Mat *out, const Mat *a, const Mat *b);

// out = [a; b] (vertical concatenation). a and b must have same column count.
MATDEF void mat_vcat(Mat *out, const Mat *a, const Mat *b);

// Extract row as column vector. Allocates new vector.
MATDEF Vec *mat_row(const Mat *m, size_t row);

// Extract column as column vector. Allocates new vector.
MATDEF Vec *mat_col(const Mat *m, size_t col);

// Return a view of row as Vec (no allocation, no copy).
// The returned Vec points to data owned by m; do not free it.
MATDEF Vec mat_row_view(const Mat *m, size_t row);

// Extract submatrix m[row_start:row_end, col_start:col_end]. Allocates new
// matrix. Indices are inclusive start, exclusive end.
MATDEF Mat *mat_slice(const Mat *m, size_t row_start, size_t row_end,
                      size_t col_start, size_t col_end);

// Copy src into m starting at (row_start, col_start).
MATDEF void mat_slice_set(Mat *m, size_t row_start, size_t col_start,
                          const Mat *src);

// Diagonal Operations

// Extract main diagonal as column vector. Allocates new vector.
MATDEF Vec *mat_diag(const Mat *m);

// Create diagonal matrix from values. Returns dim x dim matrix.
MATDEF Mat *mat_diag_from(size_t dim, const mat_elem_t *values);

// Reduction Operations

// Return sum of all elements. SIMD-optimized.
MATDEF mat_elem_t mat_sum(const Mat *a);

// Return mean of all elements.
MATDEF mat_elem_t mat_mean(const Mat *a);

// Return minimum element value. SIMD-optimized.
MATDEF mat_elem_t mat_min(const Mat *a);

// Return maximum element value. SIMD-optimized.
MATDEF mat_elem_t mat_max(const Mat *a);

// Sum along axis. axis=0: sum columns (out has rows elements).
// axis=1: sum rows (out has cols elements).
MATDEF void mat_sum_axis(Vec *out, const Mat *a, int axis);

// Return flat index of minimum element.
MATDEF size_t mat_argmin(const Mat *a);

// Return flat index of maximum element.
MATDEF size_t mat_argmax(const Mat *a);

// Return population standard deviation.
MATDEF mat_elem_t mat_std(const Mat *a);

// Norms

// General p-norm: (sum |a_i|^p)^(1/p). Uses pow(), slow for large matrices.
MATDEF mat_elem_t mat_norm(const Mat *a, size_t p);

// L2 norm. Alias for mat_norm_fro.
MATDEF mat_elem_t mat_norm2(const Mat *a);

// Infinity norm: max |a_ij|. SIMD-optimized.
MATDEF mat_elem_t mat_norm_max(const Mat *a);

// Frobenius norm: sqrt(sum a_ij^2). SIMD-optimized.
// For float32: accumulates in double to prevent overflow/underflow.
// For float64: no overflow protection (same as fast). Blue's scaling may be
// added in the future to handle extreme values.
MATDEF mat_elem_t mat_norm_fro(const Mat *a);

// Frobenius norm, fast version. SIMD-optimized.
// For float32: ~2x faster than safe, but no overflow protection.
// For float64: same as safe (no higher precision available).
// Overflows if any |a_ij|^2 exceeds type max (~1e19 for float, ~1e154 for
// double).
MATDEF mat_elem_t mat_norm_fro_fast(const Mat *a);

// Matrix Properties

// Return trace (sum of diagonal elements). Matrix must be square.
MATDEF mat_elem_t mat_trace(const Mat *a);

// Return determinant. Matrix must be square. Uses LU decomposition.
MATDEF mat_elem_t mat_det(const Mat *a);

// Return count of non-zero elements.
MATDEF mat_elem_t mat_nnz(const Mat *a);

// Decomposition

// QR decomposition via Householder reflections.
// A = Q * R where Q is orthogonal (m x m), R is upper triangular (m x n).
// Q and R must be pre-allocated with correct dimensions.
MATDEF void mat_qr(const Mat *A, Mat *Q, Mat *R);

// QR decomposition (R factor only) - faster when Q is not needed.
// Useful for least squares, rank determination, etc.
// R must be pre-allocated with dimensions (m x n).
MATDEF void mat_qr_r(const Mat *A, Mat *R);

// Householder reflection: compute v and tau such that H*x = beta*e1
// where H = I - tau*v*v^T is orthogonal.
// v is modified in-place from x (v[0] = 1, rest normalized).
// Returns beta (the resulting first element after reflection).
MATDEF mat_elem_t mat_householder(Vec *v, mat_elem_t *tau, const Vec *x);

// Apply Householder reflection from left: A = H*A = A - tau*v*(v^T*A)
// v[0] is assumed to be 1 (implicit).
MATDEF void mat_householder_left(Mat *A, const Vec *v, mat_elem_t tau);

// Apply Householder reflection from right: A = A*H = A - tau*(A*v)*v^T
// v[0] is assumed to be 1 (implicit).
MATDEF void mat_householder_right(Mat *A, const Vec *v, mat_elem_t tau);

// LU decomposition with full pivoting.
// P * A * Q = L * U where P, Q are permutations (full pivoting).
// L is lower triangular with 1s on diagonal, U is upper triangular.
// L and U must be pre-allocated with dimensions n x n.
// P and Q must be pre-allocated permutations of size n.
// Returns the number of row+column swaps (useful for determinant sign).
MATDEF int mat_lu(const Mat *A, Mat *L, Mat *U, Perm *p, Perm *q);

// P * A = L * U where P is row permutation (partial pivoting).
// Faster than mat_lu, sufficient for determinant, solve, and inverse.
// Returns the number of row swaps (useful for determinant sign).
MATDEF int mat_plu(const Mat *A, Mat *L, Mat *U, Perm *p);

// Cholesky decomposition (A = L * L^T, A must be symmetric positive definite).
// Returns 0 on success, -1 if matrix is not positive definite.
MATDEF int mat_chol(const Mat *A, Mat *L);

// Singular value decomposition (A = U * S * Vt).
// Uses one-sided Jacobi algorithm.
// U is m x m orthogonal, S is min(m,n) vector of singular values (descending),
// Vt is n x n orthogonal (V transposed).
MATDEF void mat_svd(const Mat *A, Mat *U, Vec *S, Mat *Vt);

// Matrix inverse using LU decomposition.
MATDEF void mat_inv(Mat *out, const Mat *A);

// Moore-Penrose pseudoinverse via SVD.
// out must be n x m for input A of size m x n.
// Tolerance for rank determination: max(m,n) * max(S) * epsilon.
MATDEF void mat_pinv(Mat *out, const Mat *A);

// Matrix rank via SVD.
// Returns the number of singular values above tolerance.
// Tolerance: max(m,n) * max(S) * epsilon.
MATDEF size_t mat_rank(const Mat *A);

// Condition number via SVD.
// Returns sigma_max / sigma_min.
// For singular matrices, returns infinity.
MATDEF mat_elem_t mat_cond(const Mat *A);

// Eigendecomposition.
MAT_NOT_IMPLEMENTED MATDEF void mat_eig(const Mat *A, Vec *eigenvalues,
                                        Mat *eigenvectors);

// Eigenvalues only (faster, no eigenvectors).
MAT_NOT_IMPLEMENTED MATDEF void mat_eigvals(Vec *out, const Mat *A);

// Solve Ax = b for x. A must be square and non-singular.
// Uses LU decomposition with partial pivoting.
MATDEF void mat_solve(Vec *x, const Mat *A, const Vec *b);

// Solve Ax = b for x where A is symmetric positive definite.
// Uses Cholesky decomposition (~2x faster than mat_solve for SPD matrices).
// Returns 0 on success, -1 if A is not positive definite.
MATDEF int mat_solve_spd(Vec *x, const Mat *A, const Vec *b);

// Triangular solvers (TRSV operations).
// Solve Lx = b where L is lower triangular.
MATDEF void mat_solve_tril(Vec *x, const Mat *L, const Vec *b);
// Solve Lx = b where L is unit lower triangular (implicit 1s on diagonal).
MATDEF void mat_solve_tril_unit(Vec *x, const Mat *L, const Vec *b);
// Solve Ux = b where U is upper triangular.
MATDEF void mat_solve_triu(Vec *x, const Mat *U, const Vec *b);
// Solve L^T x = b where L is lower triangular (uses L directly, no transpose).
MATDEF void mat_solve_trilt(Vec *x, const Mat *L, const Vec *b);

// Least squares solution.
MAT_NOT_IMPLEMENTED MATDEF void mat_lstsq(Vec *x, const Mat *A, const Vec *b);

// Kronecker product.
MAT_NOT_IMPLEMENTED MATDEF void mat_kron(Mat *out, const Mat *A, const Mat *B);

// 2D convolution.
MAT_NOT_IMPLEMENTED MATDEF void mat_conv2d(Mat *out, const Mat *A,
                                           const Mat *kernel);

#ifdef __cplusplus
}
#endif

#endif // MAT_H_

#ifdef MAT_IMPLEMENTATION

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Scratch arena for temporary allocations
#ifndef MAT_NO_SCRATCH
static MatArena mat_scratch_ = {0};

static inline void mat_scratch_init_(void) {
  if (mat_scratch_.buf == NULL) {
    mat_scratch_.buf = (char *)MAT_MALLOC(MAT_SCRATCH_SIZE);
    mat_scratch_.size = MAT_SCRATCH_SIZE;
    mat_scratch_.offset = 0;
  }
}

static inline void *mat_scratch_alloc_(size_t bytes) {
  mat_scratch_init_();
  bytes = (bytes + 15) & ~15; // align to 16 bytes for NEON
  if (mat_scratch_.offset + bytes > mat_scratch_.size) {
    // Fall back to heap if arena is full
    return MAT_MALLOC(bytes);
  }
  void *ptr = mat_scratch_.buf + mat_scratch_.offset;
  mat_scratch_.offset += bytes;
  return ptr;
}

static inline void mat_scratch_reset_(void) { mat_scratch_.offset = 0; }
#else
// No scratch arena - use malloc/free directly
static inline void *mat_scratch_alloc_(size_t bytes) {
  return MAT_MALLOC(bytes);
}
static inline void mat_scratch_free_(void *ptr) { MAT_FREE(ptr); }
#endif

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

  // Input values are always in row-major order (C arrays)
  // Storage depends on MAT_COLUMN_MAJOR flag
  for (size_t i = 0; i < out->rows; i++) {
    for (size_t j = 0; j < out->cols; j++) {
      MAT_SET(out, i, j, values[i * out->cols + j]);
    }
  }
}

MATDEF void mat_free_mat(Mat *m) {
  MAT_ASSERT_MAT(m);
  MAT_FREE_MAT(m);
}

MATDEF Mat *mat_zeros(size_t rows, size_t cols) { return mat_mat(rows, cols); }

MATDEF Mat *mat_ones(size_t rows, size_t cols) {
  MAT_ASSERT_DIM(rows, cols);

  Mat *result = mat_mat(rows, cols);

  mat_fill(result, (mat_elem_t)1.0);

  return result;
}

MATDEF void mat_fill(Mat *out, mat_elem_t value) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT(value >= 0);

  size_t len = out->rows * out->cols;
  for (size_t i = 0; i < len; i++)
    out->data[i] = value;
}

MATDEF void mat_eye(Mat *out) {
  MAT_ASSERT_SQUARE(out);
  size_t dim = out->rows;

  memset(out->data, 0, dim * dim * sizeof(mat_elem_t));
  for (size_t i = 0; i < dim; i++) {
    MAT_SET(out, i, i, 1);
  }
}

MATDEF Mat *mat_reye(size_t dim) {
  MAT_ASSERT(dim > 0);
  Mat *result = mat_mat(dim, dim);

  for (size_t i = 0; i < dim; i++) {
    MAT_SET(result, i, i, 1);
  }

  return result;
}

MATDEF Vec *mat_vec(size_t dim) {
  Vec *vec = mat_mat(dim, 1);
  return vec;
}

MATDEF Vec *mat_row_vec(size_t dim) {
  Vec *vec = mat_mat(1, dim);
  return vec;
}

MATDEF Vec *mat_vec_from(size_t dim, const mat_elem_t *values) {
  Vec *result = mat_from(dim, 1, values);

  return result;
}

MATDEF Perm *mat_perm(size_t n) {
  Perm *p = (Perm *)MAT_MALLOC(sizeof(Perm));
  p->data = (size_t *)MAT_MALLOC(n * sizeof(size_t));
  p->size = n;
  return p;
}

MATDEF void mat_free_perm(Perm *p) {
  MAT_FREE(p->data);
  MAT_FREE(p);
}

MATDEF void mat_perm_identity(Perm *p) {
  for (size_t i = 0; i < p->size; i++) {
    p->data[i] = i;
  }
}

MATDEF Mat *mat_perm_mat(const Perm *p) {
  size_t n = p->size;
  Mat *m = mat_zeros(n, n);
  for (size_t i = 0; i < n; i++) {
    MAT_SET(m, i, p->data[i], 1);
  }
  return m;
}

MATDEF Mat *mat_copy(const Mat *m) {
  MAT_ASSERT_MAT(m);

  Mat *result = mat_mat(m->rows, m->cols);

  return result;
}

MATDEF void mat_deep_copy(Mat *out, const Mat *src) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(src);
  MAT_ASSERT(out->rows == src->rows && out->cols == src->cols);

  size_t len = src->rows * src->cols;
  memcpy(out->data, src->data, len * sizeof(mat_elem_t));
}

MATDEF Mat *mat_rdeep_copy(const Mat *m) {
  MAT_ASSERT_MAT(m);

  Mat *result = mat_copy(m);
  size_t len = m->rows * m->cols;

  memcpy(result->data, m->data, len * sizeof(mat_elem_t));

  return result;
}

// Accessors & Info

MATDEF mat_elem_t mat_at(const Mat *m, size_t row, size_t col) {
  MAT_ASSERT_MAT(m);
  return MAT_AT(m, row, col);
}

MATDEF void mat_set_at(Mat *m, size_t row, size_t col, mat_elem_t value) {
  MAT_ASSERT_MAT(m);
  MAT_SET(m, row, col, value);
}

MATDEF MatSize mat_size(const Mat *m) {
  MatSize size = {m->rows, m->cols};
  return size;
}

MATDEF void mat_print(const Mat *mat) {
  MAT_ASSERT(mat != NULL);
  MAT_ASSERT(mat->data != NULL);

  // Always print in row-major order for human readability
  printf("[");
  for (size_t i = 0; i < mat->rows; i++) {
    if (i > 0)
      printf(" ");
    for (size_t j = 0; j < mat->cols; j++) {
      printf("%g", MAT_AT(mat, i, j));
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
MAT_INTERNAL_STATIC bool mat_equals_tol_neon_(const Mat *a, const Mat *b,
                                              mat_elem_t epsilon) {
  size_t n = a->rows * a->cols;
  const mat_elem_t *pa = a->data;
  const mat_elem_t *pb = b->data;
  MAT_NEON_TYPE eps = MAT_NEON_DUP(epsilon);

  size_t i = 0;
  for (; i + MAT_NEON_WIDTH * 4 <= n; i += MAT_NEON_WIDTH * 4) {
    MAT_NEON_TYPE diff0 =
        MAT_NEON_ABD(MAT_NEON_LOAD(pa + i), MAT_NEON_LOAD(pb + i));
    MAT_NEON_TYPE diff1 = MAT_NEON_ABD(MAT_NEON_LOAD(pa + i + MAT_NEON_WIDTH),
                                       MAT_NEON_LOAD(pb + i + MAT_NEON_WIDTH));
    MAT_NEON_TYPE diff2 =
        MAT_NEON_ABD(MAT_NEON_LOAD(pa + i + MAT_NEON_WIDTH * 2),
                     MAT_NEON_LOAD(pb + i + MAT_NEON_WIDTH * 2));
    MAT_NEON_TYPE diff3 =
        MAT_NEON_ABD(MAT_NEON_LOAD(pa + i + MAT_NEON_WIDTH * 3),
                     MAT_NEON_LOAD(pb + i + MAT_NEON_WIDTH * 3));

    MAT_NEON_UTYPE gt0 = MAT_NEON_CGT(diff0, eps);
    MAT_NEON_UTYPE gt1 = MAT_NEON_CGT(diff1, eps);
    MAT_NEON_UTYPE gt2 = MAT_NEON_CGT(diff2, eps);
    MAT_NEON_UTYPE gt3 = MAT_NEON_CGT(diff3, eps);

    MAT_NEON_UTYPE any01 = MAT_NEON_ORR_U(gt0, gt1);
    MAT_NEON_UTYPE any23 = MAT_NEON_ORR_U(gt2, gt3);
    MAT_NEON_UTYPE any = MAT_NEON_ORR_U(any01, any23);

    if (MAT_NEON_MAXV_U(any) != 0)
      return false;
  }

  for (; i + MAT_NEON_WIDTH <= n; i += MAT_NEON_WIDTH) {
    MAT_NEON_TYPE diff =
        MAT_NEON_ABD(MAT_NEON_LOAD(pa + i), MAT_NEON_LOAD(pb + i));
    MAT_NEON_UTYPE gt = MAT_NEON_CGT(diff, eps);
    if (MAT_NEON_MAXV_U(gt) != 0)
      return false;
  }

  for (; i < n; i++) {
    mat_elem_t diff = pa[i] - pb[i];
    if (diff < 0)
      diff = -diff;
    if (diff > epsilon)
      return false;
  }

  return true;
}
#endif

MAT_INTERNAL_STATIC bool mat_equals_tol_scalar_(const Mat *a, const Mat *b,
                                                mat_elem_t epsilon) {
  size_t n = a->rows * a->cols;
  for (size_t i = 0; i < n; i++) {
    mat_elem_t diff = a->data[i] - b->data[i];
    if (diff < 0)
      diff = -diff;
    if (diff > epsilon)
      return false;
  }
  return true;
}

MATDEF bool mat_equals_tol(const Mat *a, const Mat *b, mat_elem_t epsilon) {
  MAT_ASSERT_MAT(a);
  MAT_ASSERT_MAT(b);

  if (a->rows != b->rows || a->cols != b->cols)
    return false;

#ifdef MAT_HAS_ARM_NEON
  return mat_equals_tol_neon_(a, b, epsilon);
#else
  return mat_equals_tol_scalar_(a, b, epsilon);
#endif
}

/* Element-wise Unary */

MATDEF void mat_abs(Mat *out, const Mat *a) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);

  size_t len = a->rows * a->cols;
  for (size_t i = 0; i < len; i++)
    out->data[i] = fabs(a->data[i]);
}

MATDEF void mat_sqrt(Mat *out, const Mat *a) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);

  size_t len = a->rows * a->cols;
#ifdef MAT_DOUBLE_PRECISION
  for (size_t i = 0; i < len; i++)
    out->data[i] = sqrt(a->data[i]);
#else
  for (size_t i = 0; i < len; i++)
    out->data[i] = sqrtf(a->data[i]);
#endif
}

MATDEF void mat_exp(Mat *out, const Mat *a) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);

  size_t len = a->rows * a->cols;
#ifdef MAT_DOUBLE_PRECISION
  for (size_t i = 0; i < len; i++)
    out->data[i] = exp(a->data[i]);
#else
  for (size_t i = 0; i < len; i++)
    out->data[i] = expf(a->data[i]);
#endif
}

MATDEF void mat_log(Mat *out, const Mat *a) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);

  size_t len = a->rows * a->cols;
#ifdef MAT_DOUBLE_PRECISION
  for (size_t i = 0; i < len; i++)
    out->data[i] = log(a->data[i]);
#else
  for (size_t i = 0; i < len; i++)
    out->data[i] = logf(a->data[i]);
#endif
}

MATDEF void mat_log10(Mat *out, const Mat *a) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);

  size_t len = a->rows * a->cols;
#ifdef MAT_DOUBLE_PRECISION
  for (size_t i = 0; i < len; i++)
    out->data[i] = log10(a->data[i]);
#else
  for (size_t i = 0; i < len; i++)
    out->data[i] = log10f(a->data[i]);
#endif
}

MATDEF void mat_sin(Mat *out, const Mat *a) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);

  size_t len = a->rows * a->cols;
#ifdef MAT_DOUBLE_PRECISION
  for (size_t i = 0; i < len; i++)
    out->data[i] = sin(a->data[i]);
#else
  for (size_t i = 0; i < len; i++)
    out->data[i] = sinf(a->data[i]);
#endif
}

MATDEF void mat_cos(Mat *out, const Mat *a) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);

  size_t len = a->rows * a->cols;
#ifdef MAT_DOUBLE_PRECISION
  for (size_t i = 0; i < len; i++)
    out->data[i] = cos(a->data[i]);
#else
  for (size_t i = 0; i < len; i++)
    out->data[i] = cosf(a->data[i]);
#endif
}

MATDEF void mat_pow(Mat *out, const Mat *a, mat_elem_t exponent) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);

  size_t len = a->rows * a->cols;
#ifdef MAT_DOUBLE_PRECISION
  for (size_t i = 0; i < len; i++)
    out->data[i] = pow(a->data[i], exponent);
#else
  for (size_t i = 0; i < len; i++)
    out->data[i] = powf(a->data[i], exponent);
#endif
}

MATDEF void mat_clip(Mat *out, const Mat *a, mat_elem_t min_val,
                     mat_elem_t max_val) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);

  size_t len = a->rows * a->cols;
  for (size_t i = 0; i < len; i++) {
    mat_elem_t v = a->data[i];
    if (v < min_val)
      v = min_val;
    else if (v > max_val)
      v = max_val;
    out->data[i] = v;
  }
}

/* Element-wise Binary */

MATDEF void mat_div(Mat *out, const Mat *a, const Mat *b) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);
  MAT_ASSERT_MAT(b);
  MAT_ASSERT(a->rows == b->rows && a->cols == b->cols);

  size_t len = a->rows * a->cols;
  for (size_t i = 0; i < len; i++) {
    out->data[i] = a->data[i] / b->data[i];
  }
}

MATDEF void mat_atan2(Mat *out, const Mat *y, const Mat *x) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(y);
  MAT_ASSERT_MAT(x);
  MAT_ASSERT(y->rows == x->rows && y->cols == x->cols);

  size_t len = y->rows * y->cols;
#ifdef MAT_DOUBLE_PRECISION
  for (size_t i = 0; i < len; i++)
    out->data[i] = atan2(y->data[i], x->data[i]);
#else
  for (size_t i = 0; i < len; i++)
    out->data[i] = atan2f(y->data[i], x->data[i]);
#endif
}

/* Scalar Operations */

MATDEF void mat_scale(Mat *out, mat_elem_t k) {
  MAT_ASSERT_MAT(out);

  size_t len = out->rows * out->cols;
  mat_elem_t *p = out->data;
  size_t i = 0;

#ifdef MAT_HAS_ARM_NEON
  MAT_NEON_TYPE vk = MAT_NEON_DUP(k);
  for (; i + MAT_NEON_WIDTH * 4 <= len; i += MAT_NEON_WIDTH * 4) {
    MAT_NEON_TYPE v0 = MAT_NEON_LOAD(&p[i]);
    MAT_NEON_TYPE v1 = MAT_NEON_LOAD(&p[i + MAT_NEON_WIDTH]);
    MAT_NEON_TYPE v2 = MAT_NEON_LOAD(&p[i + MAT_NEON_WIDTH * 2]);
    MAT_NEON_TYPE v3 = MAT_NEON_LOAD(&p[i + MAT_NEON_WIDTH * 3]);
    MAT_NEON_STORE(&p[i], MAT_NEON_MUL(v0, vk));
    MAT_NEON_STORE(&p[i + MAT_NEON_WIDTH], MAT_NEON_MUL(v1, vk));
    MAT_NEON_STORE(&p[i + MAT_NEON_WIDTH * 2], MAT_NEON_MUL(v2, vk));
    MAT_NEON_STORE(&p[i + MAT_NEON_WIDTH * 3], MAT_NEON_MUL(v3, vk));
  }
  for (; i + MAT_NEON_WIDTH <= len; i += MAT_NEON_WIDTH) {
    MAT_NEON_TYPE v = MAT_NEON_LOAD(&p[i]);
    MAT_NEON_STORE(&p[i], MAT_NEON_MUL(v, vk));
  }
#endif

  for (; i < len; i++) {
    p[i] *= k;
  }
}

MATDEF Mat *mat_rscale(const Mat *m, mat_elem_t k) {
  Mat *result = mat_rdeep_copy(m);

  mat_scale(result, k);

  return result;
}

MATDEF mat_elem_t mat_normalize(Mat *v) {
  MAT_ASSERT_MAT(v);

  mat_elem_t norm = mat_norm2(v);
  if (norm > MAT_DEFAULT_EPSILON) {
    mat_scale(v, 1 / norm);
  }
  return norm;
}

MATDEF void mat_add_scalar(Mat *out, mat_elem_t k) {
  MAT_ASSERT_MAT(out);

  for (size_t i = 0; i < out->rows * out->cols; i++) {
    out->data[i] += k;
  }
}

MATDEF Mat *mat_radd_scalar(const Mat *m, mat_elem_t k) {
  Mat *result = mat_rdeep_copy(m);

  mat_add_scalar(result, k);

  return result;
}

/* Matrix Arithmetic */

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
    Mat *m = va_arg(args, Mat *);
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

  Mat *first = va_arg(args, Mat *);
  MAT_ASSERT_MAT(first);

  Mat *result = mat_rdeep_copy(first);

  for (size_t i = 1; i < count; i++) {
    Mat *m = va_arg(args, Mat *);
    MAT_ASSERT_MAT(m);
    MAT_ASSERT(m->rows == result->rows && m->cols == result->cols);

    for (size_t j = 0; j < result->rows * result->cols; j++) {
      result->data[j] += m->data[j];
    }
  }

  va_end(args);
  return result;
}

/* Matrix Products */

MATDEF void mat_mul(Mat *out, const Mat *a, const Mat *b) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);
  MAT_ASSERT_MAT(b);
  MAT_ASSERT(a->cols == b->rows);

  // C = 1.0 * A * B + 0.0 * C
  mat_gemm(out, 1, a, b, 0);
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
MAT_INTERNAL_STATIC mat_elem_t mat_dot_neon_(const Vec *v1, const Vec *v2) {
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

MAT_INTERNAL_STATIC mat_elem_t mat_dot_scalar_(const Vec *v1, const Vec *v2) {
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
  return mat_dot_neon_(v1, v2);
#else
  return mat_dot_scalar_(v1, v2);
#endif
}

MATDEF void mat_cross(Vec *out, const Vec *v1, const Vec *v2) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(v1);
  MAT_ASSERT_MAT(v2);
  MAT_ASSERT(v1->rows * v1->cols == 3);
  MAT_ASSERT(v2->rows * v2->cols == 3);
  MAT_ASSERT(out->rows * out->cols == 3);

  const mat_elem_t *a = v1->data;
  const mat_elem_t *b = v2->data;
  mat_elem_t *c = out->data;

  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void mat_outer_neon_(Mat *out, const Vec *v1,
                                         const Vec *v2) {
  size_t m = v1->rows * v1->cols;
  size_t n = v2->rows * v2->cols;

  const mat_elem_t *a = v1->data;
  const mat_elem_t *b = v2->data;
  mat_elem_t *c = out->data;

// Column-major: iterate over columns, store columns contiguously
  for (size_t j = 0; j < n; j++) {
    MAT_NEON_TYPE vb = MAT_NEON_DUP(b[j]);
    mat_elem_t *col = &c[j * m];
    size_t i = 0;

    for (; i + MAT_NEON_WIDTH * 4 <= m; i += MAT_NEON_WIDTH * 4) {
      MAT_NEON_TYPE va0 = MAT_NEON_LOAD(&a[i]);
      MAT_NEON_TYPE va1 = MAT_NEON_LOAD(&a[i + MAT_NEON_WIDTH]);
      MAT_NEON_TYPE va2 = MAT_NEON_LOAD(&a[i + MAT_NEON_WIDTH * 2]);
      MAT_NEON_TYPE va3 = MAT_NEON_LOAD(&a[i + MAT_NEON_WIDTH * 3]);
      MAT_NEON_STORE(&col[i], MAT_NEON_MUL(va0, vb));
      MAT_NEON_STORE(&col[i + MAT_NEON_WIDTH], MAT_NEON_MUL(va1, vb));
      MAT_NEON_STORE(&col[i + MAT_NEON_WIDTH * 2], MAT_NEON_MUL(va2, vb));
      MAT_NEON_STORE(&col[i + MAT_NEON_WIDTH * 3], MAT_NEON_MUL(va3, vb));
    }

    for (; i + MAT_NEON_WIDTH <= m; i += MAT_NEON_WIDTH) {
      MAT_NEON_TYPE va = MAT_NEON_LOAD(&a[i]);
      MAT_NEON_STORE(&col[i], MAT_NEON_MUL(va, vb));
    }

    for (; i < m; i++) {
      col[i] = a[i] * b[j];
    }
  }
}
#endif

MAT_INTERNAL_STATIC void mat_outer_scalar_(Mat *out, const Vec *v1,
                                           const Vec *v2) {
  size_t m = v1->rows * v1->cols;
  size_t n = v2->rows * v2->cols;

  const mat_elem_t *a = v1->data;
  const mat_elem_t *b = v2->data;

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      MAT_SET(out, i, j, a[i] * b[j]);
    }
  }
}

MATDEF void mat_outer(Mat *out, const Vec *v1, const Vec *v2) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(v1);
  MAT_ASSERT_MAT(v2);

  size_t m = v1->rows * v1->cols;
  size_t n = v2->rows * v2->cols;
  MAT_ASSERT(out->rows == m && out->cols == n);

#ifdef MAT_HAS_ARM_NEON
  mat_outer_neon_(out, v1, v2);
#else
  mat_outer_scalar_(out, v1, v2);
#endif
}

MATDEF mat_elem_t mat_bilinear(const Vec *x, const Mat *A, const Vec *y) {
  MAT_ASSERT_MAT(x);
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(y);

  size_t m = x->rows * x->cols;
  size_t n = y->rows * y->cols;
  MAT_ASSERT(A->rows == m && A->cols == n);

  // temp = A * y, result = x^T * temp
  Vec *temp = mat_vec(m);
  mat_gemv(temp, 1, A, y, 0);
  mat_elem_t result = mat_dot(x, temp);
  mat_free_mat(temp);
  return result;
}

MATDEF mat_elem_t mat_quadform(const Vec *x, const Mat *A) {
  MAT_ASSERT_MAT(x);
  MAT_ASSERT_MAT(A);
  MAT_ASSERT(A->rows == A->cols);

  size_t n = x->rows * x->cols;
  MAT_ASSERT(A->rows == n);

  return mat_bilinear(x, A, x);
}

/* Fused Operations (BLAS-like) */

// y += alpha * x (BLAS Level-1: axpy)
#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void mat_axpy_neon_(Vec *y, mat_elem_t alpha,
                                        const Vec *x) {
  size_t n = x->rows;
  mat_elem_t *py = y->data;
  const mat_elem_t *px = x->data;
  MAT_NEON_TYPE valpha = MAT_NEON_DUP(alpha);

  size_t i = 0;
  for (; i + MAT_NEON_WIDTH * 4 <= n; i += MAT_NEON_WIDTH * 4) {
    MAT_NEON_TYPE vy0 = MAT_NEON_LOAD(&py[i]);
    MAT_NEON_TYPE vy1 = MAT_NEON_LOAD(&py[i + MAT_NEON_WIDTH]);
    MAT_NEON_TYPE vy2 = MAT_NEON_LOAD(&py[i + MAT_NEON_WIDTH * 2]);
    MAT_NEON_TYPE vy3 = MAT_NEON_LOAD(&py[i + MAT_NEON_WIDTH * 3]);
    MAT_NEON_TYPE vx0 = MAT_NEON_LOAD(&px[i]);
    MAT_NEON_TYPE vx1 = MAT_NEON_LOAD(&px[i + MAT_NEON_WIDTH]);
    MAT_NEON_TYPE vx2 = MAT_NEON_LOAD(&px[i + MAT_NEON_WIDTH * 2]);
    MAT_NEON_TYPE vx3 = MAT_NEON_LOAD(&px[i + MAT_NEON_WIDTH * 3]);
    vy0 = MAT_NEON_FMA(vy0, vx0, valpha);
    vy1 = MAT_NEON_FMA(vy1, vx1, valpha);
    vy2 = MAT_NEON_FMA(vy2, vx2, valpha);
    vy3 = MAT_NEON_FMA(vy3, vx3, valpha);
    MAT_NEON_STORE(&py[i], vy0);
    MAT_NEON_STORE(&py[i + MAT_NEON_WIDTH], vy1);
    MAT_NEON_STORE(&py[i + MAT_NEON_WIDTH * 2], vy2);
    MAT_NEON_STORE(&py[i + MAT_NEON_WIDTH * 3], vy3);
  }

  for (; i + MAT_NEON_WIDTH <= n; i += MAT_NEON_WIDTH) {
    MAT_NEON_TYPE vy = MAT_NEON_LOAD(&py[i]);
    MAT_NEON_TYPE vx = MAT_NEON_LOAD(&px[i]);
    vy = MAT_NEON_FMA(vy, vx, valpha);
    MAT_NEON_STORE(&py[i], vy);
  }

  for (; i < n; i++) {
    py[i] += alpha * px[i];
  }
}
#endif

MAT_INTERNAL_STATIC void mat_axpy_scalar_(Vec *y, mat_elem_t alpha,
                                          const Vec *x) {
  size_t n = x->rows;
  for (size_t i = 0; i < n; i++) {
    y->data[i] += alpha * x->data[i];
  }
}

MATDEF void mat_axpy(Vec *y, mat_elem_t alpha, const Vec *x) {
  MAT_ASSERT_MAT(y);
  MAT_ASSERT_MAT(x);
  MAT_ASSERT(y->rows == x->rows);

#ifdef MAT_HAS_ARM_NEON
  mat_axpy_neon_(y, alpha, x);
#else
  mat_axpy_scalar_(y, alpha, x);
#endif
}

// y = alpha * A * x + beta * y (BLAS Level-2: gemv)
#ifdef MAT_HAS_ARM_NEON
// Forward declaration for fallback
MAT_INTERNAL_STATIC void mat_gemv_scalar_(Vec *y, mat_elem_t alpha,
                                          const Mat *A, const Vec *x,
                                          mat_elem_t beta);

MAT_INTERNAL_STATIC void mat_gemv_neon_(Vec *y, mat_elem_t alpha, const Mat *A,
                                        const Vec *x, mat_elem_t beta) {
  // Column-major: columns are contiguous, use AXPY-style algorithm
  // Process 8 columns at a time for better instruction-level parallelism
  size_t m = A->rows;
  size_t n = A->cols;
  mat_elem_t *py = y->data;
  const mat_elem_t *pa = A->data;
  const mat_elem_t *px = x->data;

  // Scale y by beta first
  if (beta == 0) {
    for (size_t i = 0; i < m; i++)
      py[i] = 0;
  } else if (beta != 1) {
    for (size_t i = 0; i < m; i++)
      py[i] *= beta;
  }

  // Process 8 columns at a time for better ILP
  size_t j = 0;
  for (; j + 8 <= n; j += 8) {
    MAT_NEON_TYPE vx0 = MAT_NEON_DUP(alpha * px[j]);
    MAT_NEON_TYPE vx1 = MAT_NEON_DUP(alpha * px[j + 1]);
    MAT_NEON_TYPE vx2 = MAT_NEON_DUP(alpha * px[j + 2]);
    MAT_NEON_TYPE vx3 = MAT_NEON_DUP(alpha * px[j + 3]);
    MAT_NEON_TYPE vx4 = MAT_NEON_DUP(alpha * px[j + 4]);
    MAT_NEON_TYPE vx5 = MAT_NEON_DUP(alpha * px[j + 5]);
    MAT_NEON_TYPE vx6 = MAT_NEON_DUP(alpha * px[j + 6]);
    MAT_NEON_TYPE vx7 = MAT_NEON_DUP(alpha * px[j + 7]);

    size_t i = 0;
    for (; i + MAT_NEON_WIDTH * 2 <= m; i += MAT_NEON_WIDTH * 2) {
      MAT_NEON_TYPE vy0 = MAT_NEON_LOAD(&py[i]);
      MAT_NEON_TYPE vy1 = MAT_NEON_LOAD(&py[i + MAT_NEON_WIDTH]);

      vy0 = MAT_NEON_FMA(vy0, MAT_NEON_LOAD(&pa[(j + 0) * m + i]), vx0);
      vy1 = MAT_NEON_FMA(vy1, MAT_NEON_LOAD(&pa[(j + 0) * m + i + MAT_NEON_WIDTH]), vx0);
      vy0 = MAT_NEON_FMA(vy0, MAT_NEON_LOAD(&pa[(j + 1) * m + i]), vx1);
      vy1 = MAT_NEON_FMA(vy1, MAT_NEON_LOAD(&pa[(j + 1) * m + i + MAT_NEON_WIDTH]), vx1);
      vy0 = MAT_NEON_FMA(vy0, MAT_NEON_LOAD(&pa[(j + 2) * m + i]), vx2);
      vy1 = MAT_NEON_FMA(vy1, MAT_NEON_LOAD(&pa[(j + 2) * m + i + MAT_NEON_WIDTH]), vx2);
      vy0 = MAT_NEON_FMA(vy0, MAT_NEON_LOAD(&pa[(j + 3) * m + i]), vx3);
      vy1 = MAT_NEON_FMA(vy1, MAT_NEON_LOAD(&pa[(j + 3) * m + i + MAT_NEON_WIDTH]), vx3);
      vy0 = MAT_NEON_FMA(vy0, MAT_NEON_LOAD(&pa[(j + 4) * m + i]), vx4);
      vy1 = MAT_NEON_FMA(vy1, MAT_NEON_LOAD(&pa[(j + 4) * m + i + MAT_NEON_WIDTH]), vx4);
      vy0 = MAT_NEON_FMA(vy0, MAT_NEON_LOAD(&pa[(j + 5) * m + i]), vx5);
      vy1 = MAT_NEON_FMA(vy1, MAT_NEON_LOAD(&pa[(j + 5) * m + i + MAT_NEON_WIDTH]), vx5);
      vy0 = MAT_NEON_FMA(vy0, MAT_NEON_LOAD(&pa[(j + 6) * m + i]), vx6);
      vy1 = MAT_NEON_FMA(vy1, MAT_NEON_LOAD(&pa[(j + 6) * m + i + MAT_NEON_WIDTH]), vx6);
      vy0 = MAT_NEON_FMA(vy0, MAT_NEON_LOAD(&pa[(j + 7) * m + i]), vx7);
      vy1 = MAT_NEON_FMA(vy1, MAT_NEON_LOAD(&pa[(j + 7) * m + i + MAT_NEON_WIDTH]), vx7);

      MAT_NEON_STORE(&py[i], vy0);
      MAT_NEON_STORE(&py[i + MAT_NEON_WIDTH], vy1);
    }
    for (; i < m; i++) {
      py[i] += alpha * (pa[(j + 0) * m + i] * px[j + 0] +
                        pa[(j + 1) * m + i] * px[j + 1] +
                        pa[(j + 2) * m + i] * px[j + 2] +
                        pa[(j + 3) * m + i] * px[j + 3] +
                        pa[(j + 4) * m + i] * px[j + 4] +
                        pa[(j + 5) * m + i] * px[j + 5] +
                        pa[(j + 6) * m + i] * px[j + 6] +
                        pa[(j + 7) * m + i] * px[j + 7]);
    }
  }

  // Handle remaining 4 columns
  for (; j + 4 <= n; j += 4) {
    MAT_NEON_TYPE vx0 = MAT_NEON_DUP(alpha * px[j]);
    MAT_NEON_TYPE vx1 = MAT_NEON_DUP(alpha * px[j + 1]);
    MAT_NEON_TYPE vx2 = MAT_NEON_DUP(alpha * px[j + 2]);
    MAT_NEON_TYPE vx3 = MAT_NEON_DUP(alpha * px[j + 3]);

    size_t i = 0;
    for (; i + MAT_NEON_WIDTH <= m; i += MAT_NEON_WIDTH) {
      MAT_NEON_TYPE vy = MAT_NEON_LOAD(&py[i]);
      vy = MAT_NEON_FMA(vy, MAT_NEON_LOAD(&pa[(j + 0) * m + i]), vx0);
      vy = MAT_NEON_FMA(vy, MAT_NEON_LOAD(&pa[(j + 1) * m + i]), vx1);
      vy = MAT_NEON_FMA(vy, MAT_NEON_LOAD(&pa[(j + 2) * m + i]), vx2);
      vy = MAT_NEON_FMA(vy, MAT_NEON_LOAD(&pa[(j + 3) * m + i]), vx3);
      MAT_NEON_STORE(&py[i], vy);
    }
    for (; i < m; i++) {
      py[i] += alpha * (pa[(j + 0) * m + i] * px[j + 0] +
                        pa[(j + 1) * m + i] * px[j + 1] +
                        pa[(j + 2) * m + i] * px[j + 2] +
                        pa[(j + 3) * m + i] * px[j + 3]);
    }
  }

  // Handle remaining columns
  for (; j < n; j++) {
    const mat_elem_t *col = &pa[j * m];
    mat_elem_t axj = alpha * px[j];
    for (size_t i = 0; i < m; i++) {
      py[i] += col[i] * axj;
    }
  }
}
#endif

MAT_INTERNAL_STATIC void mat_gemv_scalar_(Vec *y, mat_elem_t alpha,
                                          const Mat *A, const Vec *x,
                                          mat_elem_t beta) {
  size_t m = A->rows;
  size_t n = A->cols;

  for (size_t i = 0; i < m; i++) {
    mat_elem_t sum = 0;
    for (size_t j = 0; j < n; j++) {
      sum += MAT_AT(A, i, j) * x->data[j];
    }
    y->data[i] = alpha * sum + beta * y->data[i];
  }
}

MATDEF void mat_gemv(Vec *y, mat_elem_t alpha, const Mat *A, const Vec *x,
                     mat_elem_t beta) {
  MAT_ASSERT_MAT(y);
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(x);
  MAT_ASSERT(A->rows == y->rows);
  MAT_ASSERT(A->cols == x->rows);

#ifdef MAT_HAS_ARM_NEON
  mat_gemv_neon_(y, alpha, A, x, beta);
#else
  mat_gemv_scalar_(y, alpha, A, x, beta);
#endif
}

// y = alpha * A^T * x + beta * y (GEMV transposed)
// A is m×n, x is m×1, y is n×1
#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void mat_gemv_t_neon_(Vec *y, mat_elem_t alpha,
                                          const Mat *A, const Vec *x,
                                          mat_elem_t beta) {
  size_t m = A->rows;
  size_t n = A->cols;
  mat_elem_t *py = y->data;
  const mat_elem_t *pa = A->data;
  const mat_elem_t *px = x->data;

// Column-major: y[j] = alpha * dot(A[:,j], x) + beta * y[j]
  // Column j is contiguous at &pa[j * m], so this is a series of dot products
  // Process 4 columns at a time for better ILP
  size_t j = 0;
  for (; j + 4 <= n; j += 4) {
    const mat_elem_t *col0 = &pa[j * m];
    const mat_elem_t *col1 = &pa[(j + 1) * m];
    const mat_elem_t *col2 = &pa[(j + 2) * m];
    const mat_elem_t *col3 = &pa[(j + 3) * m];

    MAT_NEON_TYPE sum0 = MAT_NEON_DUP(0);
    MAT_NEON_TYPE sum1 = MAT_NEON_DUP(0);
    MAT_NEON_TYPE sum2 = MAT_NEON_DUP(0);
    MAT_NEON_TYPE sum3 = MAT_NEON_DUP(0);

    size_t i = 0;
    for (; i + MAT_NEON_WIDTH <= m; i += MAT_NEON_WIDTH) {
      MAT_NEON_TYPE vx = MAT_NEON_LOAD(&px[i]);
      sum0 = MAT_NEON_FMA(sum0, MAT_NEON_LOAD(&col0[i]), vx);
      sum1 = MAT_NEON_FMA(sum1, MAT_NEON_LOAD(&col1[i]), vx);
      sum2 = MAT_NEON_FMA(sum2, MAT_NEON_LOAD(&col2[i]), vx);
      sum3 = MAT_NEON_FMA(sum3, MAT_NEON_LOAD(&col3[i]), vx);
    }

    mat_elem_t dot0 = MAT_NEON_ADDV(sum0);
    mat_elem_t dot1 = MAT_NEON_ADDV(sum1);
    mat_elem_t dot2 = MAT_NEON_ADDV(sum2);
    mat_elem_t dot3 = MAT_NEON_ADDV(sum3);

    // Scalar remainder
    for (; i < m; i++) {
      dot0 += col0[i] * px[i];
      dot1 += col1[i] * px[i];
      dot2 += col2[i] * px[i];
      dot3 += col3[i] * px[i];
    }

    py[j] = alpha * dot0 + beta * py[j];
    py[j + 1] = alpha * dot1 + beta * py[j + 1];
    py[j + 2] = alpha * dot2 + beta * py[j + 2];
    py[j + 3] = alpha * dot3 + beta * py[j + 3];
  }

  // Handle remaining columns
  for (; j < n; j++) {
    const mat_elem_t *col = &pa[j * m];
    MAT_NEON_TYPE sum = MAT_NEON_DUP(0);

    size_t i = 0;
    for (; i + MAT_NEON_WIDTH <= m; i += MAT_NEON_WIDTH) {
      MAT_NEON_TYPE vx = MAT_NEON_LOAD(&px[i]);
      sum = MAT_NEON_FMA(sum, MAT_NEON_LOAD(&col[i]), vx);
    }

    mat_elem_t dot = MAT_NEON_ADDV(sum);
    for (; i < m; i++) {
      dot += col[i] * px[i];
    }

    py[j] = alpha * dot + beta * py[j];
  }
}
#endif

MAT_INTERNAL_STATIC void mat_gemv_t_scalar_(Vec *y, mat_elem_t alpha,
                                            const Mat *A, const Vec *x,
                                            mat_elem_t beta) {
  size_t m = A->rows;
  size_t n = A->cols;

  // y = beta * y first
  for (size_t j = 0; j < n; j++) {
    y->data[j] *= beta;
  }

  // y += alpha * A^T * x
  for (size_t i = 0; i < m; i++) {
    mat_elem_t xi = alpha * x->data[i];
    for (size_t j = 0; j < n; j++) {
      y->data[j] += MAT_AT(A, i, j) * xi;
    }
  }
}

MATDEF void mat_gemv_t(Vec *y, mat_elem_t alpha, const Mat *A, const Vec *x,
                       mat_elem_t beta) {
  MAT_ASSERT_MAT(y);
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(x);
  MAT_ASSERT(A->cols == y->rows); // A^T has n rows
  MAT_ASSERT(A->rows == x->rows); // A^T has m cols

#ifdef MAT_HAS_ARM_NEON
  mat_gemv_t_neon_(y, alpha, A, x, beta);
#else
  mat_gemv_t_scalar_(y, alpha, A, x, beta);
#endif
}

// A += alpha * x * y^T (BLAS Level-2: ger - rank-1 update)
#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void mat_ger_neon_(Mat *A, mat_elem_t alpha, const Vec *x,
                                       const Vec *y) {
  size_t m = A->rows;
  size_t n = A->cols;
  mat_elem_t *pa = A->data;
  const mat_elem_t *px = x->data;
  const mat_elem_t *py = y->data;

// Column-major: A[:,j] += alpha * y[j] * x (columns are contiguous)
  for (size_t j = 0; j < n; j++) {
    mat_elem_t *col = &pa[j * m];
    MAT_NEON_TYPE vyj = MAT_NEON_DUP(alpha * py[j]);

    size_t i = 0;
    for (; i + MAT_NEON_WIDTH * 4 <= m; i += MAT_NEON_WIDTH * 4) {
      MAT_NEON_TYPE va0 = MAT_NEON_LOAD(&col[i]);
      MAT_NEON_TYPE va1 = MAT_NEON_LOAD(&col[i + MAT_NEON_WIDTH]);
      MAT_NEON_TYPE va2 = MAT_NEON_LOAD(&col[i + MAT_NEON_WIDTH * 2]);
      MAT_NEON_TYPE va3 = MAT_NEON_LOAD(&col[i + MAT_NEON_WIDTH * 3]);
      MAT_NEON_TYPE vx0 = MAT_NEON_LOAD(&px[i]);
      MAT_NEON_TYPE vx1 = MAT_NEON_LOAD(&px[i + MAT_NEON_WIDTH]);
      MAT_NEON_TYPE vx2 = MAT_NEON_LOAD(&px[i + MAT_NEON_WIDTH * 2]);
      MAT_NEON_TYPE vx3 = MAT_NEON_LOAD(&px[i + MAT_NEON_WIDTH * 3]);
      va0 = MAT_NEON_FMA(va0, vx0, vyj);
      va1 = MAT_NEON_FMA(va1, vx1, vyj);
      va2 = MAT_NEON_FMA(va2, vx2, vyj);
      va3 = MAT_NEON_FMA(va3, vx3, vyj);
      MAT_NEON_STORE(&col[i], va0);
      MAT_NEON_STORE(&col[i + MAT_NEON_WIDTH], va1);
      MAT_NEON_STORE(&col[i + MAT_NEON_WIDTH * 2], va2);
      MAT_NEON_STORE(&col[i + MAT_NEON_WIDTH * 3], va3);
    }

    for (; i + MAT_NEON_WIDTH <= m; i += MAT_NEON_WIDTH) {
      MAT_NEON_TYPE va = MAT_NEON_LOAD(&col[i]);
      MAT_NEON_TYPE vx = MAT_NEON_LOAD(&px[i]);
      va = MAT_NEON_FMA(va, vx, vyj);
      MAT_NEON_STORE(&col[i], va);
    }

    mat_elem_t yj = alpha * py[j];
    for (; i < m; i++) {
      col[i] += px[i] * yj;
    }
  }
}
#endif

MAT_INTERNAL_STATIC void mat_ger_scalar_(Mat *A, mat_elem_t alpha, const Vec *x,
                                         const Vec *y) {
  size_t m = A->rows;
  size_t n = A->cols;

  for (size_t i = 0; i < m; i++) {
    mat_elem_t xi = alpha * x->data[i];
    for (size_t j = 0; j < n; j++) {
      MAT_SET(A, i, j, MAT_AT(A, i, j) + xi * y->data[j]);
    }
  }
}

MATDEF void mat_ger(Mat *A, mat_elem_t alpha, const Vec *x, const Vec *y) {
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(x);
  MAT_ASSERT_MAT(y);
  MAT_ASSERT(A->rows == x->rows);
  MAT_ASSERT(A->cols == y->rows);

#ifdef MAT_HAS_ARM_NEON
  mat_ger_neon_(A, alpha, x, y);
#else
  mat_ger_scalar_(A, alpha, x, y);
#endif
}

// A += alpha * x * x^T (BLAS Level-2: syr - symmetric rank-1 update)
// Lower triangle version
MAT_INTERNAL_STATIC void mat_syr_lower_scalar_(Mat *A, mat_elem_t alpha,
                                               const Vec *x) {
  size_t n = A->rows;
  for (size_t i = 0; i < n; i++) {
    mat_elem_t xi = alpha * x->data[i];
    for (size_t j = 0; j <= i; j++) {
      MAT_SET(A, i, j, MAT_AT(A, i, j) + xi * x->data[j]);
    }
  }
}

// Upper triangle version
MAT_INTERNAL_STATIC void mat_syr_upper_scalar_(Mat *A, mat_elem_t alpha,
                                               const Vec *x) {
  size_t n = A->rows;
  for (size_t i = 0; i < n; i++) {
    mat_elem_t xi = alpha * x->data[i];
    for (size_t j = i; j < n; j++) {
      MAT_SET(A, i, j, MAT_AT(A, i, j) + xi * x->data[j]);
    }
  }
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void mat_syr_lower_neon_(Mat *A, mat_elem_t alpha,
                                             const Vec *x) {
  size_t n = A->rows;
  mat_elem_t *pa = A->data;
  const mat_elem_t *px = x->data;

// Column-major: iterate by columns, A[j:n, j] += alpha * x[j] * x[j:n]
  // Column j is contiguous at &pa[j * n], lower triangle is rows j to n-1
  for (size_t j = 0; j < n; j++) {
    mat_elem_t *col = &pa[j * n];
    mat_elem_t xj = alpha * px[j];
    MAT_NEON_TYPE vxj = MAT_NEON_DUP(xj);

    size_t i = j;
    // SIMD: process full vectors
    for (; i + MAT_NEON_WIDTH <= n; i += MAT_NEON_WIDTH) {
      MAT_NEON_TYPE va = MAT_NEON_LOAD(&col[i]);
      MAT_NEON_TYPE vx = MAT_NEON_LOAD(&px[i]);
      va = MAT_NEON_FMA(va, vx, vxj);
      MAT_NEON_STORE(&col[i], va);
    }
    // Scalar remainder
    for (; i < n; i++) {
      col[i] += xj * px[i];
    }
  }
}

MAT_INTERNAL_STATIC void mat_syr_upper_neon_(Mat *A, mat_elem_t alpha,
                                             const Vec *x) {
  size_t n = A->rows;
  mat_elem_t *pa = A->data;
  const mat_elem_t *px = x->data;

// Column-major: iterate by columns, A[0:j+1, j] += alpha * x[j] * x[0:j+1]
  // Column j is contiguous at &pa[j * n], upper triangle is rows 0 to j
  for (size_t j = 0; j < n; j++) {
    mat_elem_t *col = &pa[j * n];
    mat_elem_t xj = alpha * px[j];
    MAT_NEON_TYPE vxj = MAT_NEON_DUP(xj);

    size_t i = 0;
    // SIMD: process full vectors while i + MAT_NEON_WIDTH <= j + 1
    for (; i + MAT_NEON_WIDTH <= j + 1; i += MAT_NEON_WIDTH) {
      MAT_NEON_TYPE va = MAT_NEON_LOAD(&col[i]);
      MAT_NEON_TYPE vx = MAT_NEON_LOAD(&px[i]);
      va = MAT_NEON_FMA(va, vx, vxj);
      MAT_NEON_STORE(&col[i], va);
    }
    // Scalar remainder: i <= j
    for (; i <= j; i++) {
      col[i] += xj * px[i];
    }
  }
}
#endif

MATDEF void mat_syr(Mat *A, mat_elem_t alpha, const Vec *x, char uplo) {
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(x);
  MAT_ASSERT(A->rows == A->cols);
  MAT_ASSERT(A->rows == x->rows);

  if (uplo == 'L' || uplo == 'l') {
#ifdef MAT_HAS_ARM_NEON
    mat_syr_lower_neon_(A, alpha, x);
#else
    mat_syr_lower_scalar_(A, alpha, x);
#endif
  } else {
#ifdef MAT_HAS_ARM_NEON
    mat_syr_upper_neon_(A, alpha, x);
#else
    mat_syr_upper_scalar_(A, alpha, x);
#endif
  }
}

// Unit lower triangular GEMM for QR decomposition
// V is unit lower triangular stored column-major: V[col * ldv + row] for row >
// col Implicit 1 on diagonal, implicit 0 above diagonal

#ifdef MAT_HAS_ARM_NEON
// W = V^T * R where V is unit lower triangular (panel_rows x kb), R is
// (panel_rows x N) Result W is (kb x N), stored row-major with stride ldw V
// stored column-major: V[col * ldv + row] for row > col
MAT_INTERNAL_STATIC void
mat_gemm_unit_lower_t_neon_(mat_elem_t *W, size_t ldw, const mat_elem_t *V,
                            size_t ldv, const mat_elem_t *R, size_t ldr,
                            size_t kb, size_t panel_rows, size_t N) {

  for (size_t ii = 0; ii < kb; ii++) {
    mat_elem_t *Wi = &W[ii * ldw];
    const mat_elem_t *Rii =
        &R[ii * ldr]; // Row ii of R (diagonal element of V^T)

    // Initialize W[ii,:] = R[ii,:] (the diagonal 1 in V^T)
    size_t j = 0;
    for (; j + MAT_NEON_WIDTH * 4 <= N; j += MAT_NEON_WIDTH * 4) {
      MAT_NEON_STORE(&Wi[j], MAT_NEON_LOAD(&Rii[j]));
      MAT_NEON_STORE(&Wi[j + MAT_NEON_WIDTH],
                     MAT_NEON_LOAD(&Rii[j + MAT_NEON_WIDTH]));
      MAT_NEON_STORE(&Wi[j + MAT_NEON_WIDTH * 2],
                     MAT_NEON_LOAD(&Rii[j + MAT_NEON_WIDTH * 2]));
      MAT_NEON_STORE(&Wi[j + MAT_NEON_WIDTH * 3],
                     MAT_NEON_LOAD(&Rii[j + MAT_NEON_WIDTH * 3]));
    }
    for (; j < N; j++)
      Wi[j] = Rii[j];

    // Accumulate: W[ii,:] += V[ii,r] * R[r,:] for r = ii+1 to panel_rows-1
    for (size_t r = ii + 1; r < panel_rows; r++) {
      mat_elem_t v_val = V[ii * ldv + r];
      const mat_elem_t *Rr = &R[r * ldr];
      MAT_NEON_TYPE vv = MAT_NEON_DUP(v_val);

      j = 0;
      for (; j + MAT_NEON_WIDTH * 4 <= N; j += MAT_NEON_WIDTH * 4) {
        MAT_NEON_TYPE w0 = MAT_NEON_LOAD(&Wi[j]);
        MAT_NEON_TYPE w1 = MAT_NEON_LOAD(&Wi[j + MAT_NEON_WIDTH]);
        MAT_NEON_TYPE w2 = MAT_NEON_LOAD(&Wi[j + MAT_NEON_WIDTH * 2]);
        MAT_NEON_TYPE w3 = MAT_NEON_LOAD(&Wi[j + MAT_NEON_WIDTH * 3]);
        MAT_NEON_TYPE r0 = MAT_NEON_LOAD(&Rr[j]);
        MAT_NEON_TYPE r1 = MAT_NEON_LOAD(&Rr[j + MAT_NEON_WIDTH]);
        MAT_NEON_TYPE r2 = MAT_NEON_LOAD(&Rr[j + MAT_NEON_WIDTH * 2]);
        MAT_NEON_TYPE r3 = MAT_NEON_LOAD(&Rr[j + MAT_NEON_WIDTH * 3]);
        MAT_NEON_STORE(&Wi[j], MAT_NEON_FMA(w0, vv, r0));
        MAT_NEON_STORE(&Wi[j + MAT_NEON_WIDTH], MAT_NEON_FMA(w1, vv, r1));
        MAT_NEON_STORE(&Wi[j + MAT_NEON_WIDTH * 2], MAT_NEON_FMA(w2, vv, r2));
        MAT_NEON_STORE(&Wi[j + MAT_NEON_WIDTH * 3], MAT_NEON_FMA(w3, vv, r3));
      }
      for (; j < N; j++)
        Wi[j] += v_val * Rr[j];
    }
  }
}

// C -= V * W where V is unit lower triangular (panel_rows x kb), W is (kb x N)
// C is (panel_rows x N) with stride ldc
// V stored column-major: V[col * ldv + row] for row > col
MAT_INTERNAL_STATIC void
mat_gemm_unit_lower_neon_(mat_elem_t *C, size_t ldc, const mat_elem_t *V,
                          size_t ldv, const mat_elem_t *W, size_t ldw,
                          size_t panel_rows, size_t kb, size_t N) {

  for (size_t r = 0; r < panel_rows; r++) {
    mat_elem_t *Cr = &C[r * ldc];
    size_t ii_max = (r < kb) ? r + 1 : kb;

    for (size_t ii = 0; ii < ii_max; ii++) {
      mat_elem_t v_val = (r == ii) ? 1.0f : V[ii * ldv + r];
      const mat_elem_t *Wii = &W[ii * ldw];
      MAT_NEON_TYPE vv = MAT_NEON_DUP(v_val);

      size_t j = 0;
      for (; j + MAT_NEON_WIDTH * 4 <= N; j += MAT_NEON_WIDTH * 4) {
        MAT_NEON_TYPE c0 = MAT_NEON_LOAD(&Cr[j]);
        MAT_NEON_TYPE c1 = MAT_NEON_LOAD(&Cr[j + MAT_NEON_WIDTH]);
        MAT_NEON_TYPE c2 = MAT_NEON_LOAD(&Cr[j + MAT_NEON_WIDTH * 2]);
        MAT_NEON_TYPE c3 = MAT_NEON_LOAD(&Cr[j + MAT_NEON_WIDTH * 3]);
        MAT_NEON_TYPE w0 = MAT_NEON_LOAD(&Wii[j]);
        MAT_NEON_TYPE w1 = MAT_NEON_LOAD(&Wii[j + MAT_NEON_WIDTH]);
        MAT_NEON_TYPE w2 = MAT_NEON_LOAD(&Wii[j + MAT_NEON_WIDTH * 2]);
        MAT_NEON_TYPE w3 = MAT_NEON_LOAD(&Wii[j + MAT_NEON_WIDTH * 3]);
        MAT_NEON_STORE(&Cr[j], MAT_NEON_FMS(c0, vv, w0));
        MAT_NEON_STORE(&Cr[j + MAT_NEON_WIDTH], MAT_NEON_FMS(c1, vv, w1));
        MAT_NEON_STORE(&Cr[j + MAT_NEON_WIDTH * 2], MAT_NEON_FMS(c2, vv, w2));
        MAT_NEON_STORE(&Cr[j + MAT_NEON_WIDTH * 3], MAT_NEON_FMS(c3, vv, w3));
      }
      for (; j < N; j++)
        Cr[j] -= v_val * Wii[j];
    }
  }
}

// W = Q * V where Q is (M x panel_rows) with stride ldq, V is unit lower
// (panel_rows x kb) Result W is (M x kb) with stride ldw
MAT_INTERNAL_STATIC void
mat_gemm_q_unit_lower_neon_(mat_elem_t *W, size_t ldw, const mat_elem_t *Q,
                            size_t ldq, const mat_elem_t *V, size_t ldv,
                            size_t M, size_t panel_rows, size_t kb) {

  for (size_t ii = 0; ii < M; ii++) {
    const mat_elem_t *Qi = &Q[ii * ldq];
    mat_elem_t *Wi = &W[ii * ldw];

    for (size_t jj = 0; jj < kb; jj++) {
      // W[ii, jj] = Q[ii, jj] * 1 + sum(Q[ii, r] * V[jj, r]) for r > jj
      mat_elem_t dot = Qi[jj]; // diagonal 1
      for (size_t r = jj + 1; r < panel_rows; r++) {
        dot += Qi[r] * V[jj * ldv + r];
      }
      Wi[jj] = dot;
    }
  }
}

// Q -= W * V^T where W is (M x kb), V^T is unit upper (kb x panel_rows)
// Q is (M x panel_rows) with stride ldq
MAT_INTERNAL_STATIC void
mat_gemm_sub_w_unit_upper_neon_(mat_elem_t *Q, size_t ldq, const mat_elem_t *W,
                                size_t ldw, const mat_elem_t *V, size_t ldv,
                                size_t M, size_t kb, size_t panel_rows) {

  // Process column by column of Q
  for (size_t r = 0; r < panel_rows; r++) {
    size_t jj_max = (r < kb) ? r + 1 : kb;

    for (size_t ii = 0; ii < M; ii++) {
      mat_elem_t sum = 0;
      for (size_t jj = 0; jj < jj_max; jj++) {
        mat_elem_t v_val = (r == jj) ? 1.0f : V[jj * ldv + r];
        sum += W[ii * ldw + jj] * v_val;
      }
      Q[ii * ldq + r] -= sum;
    }
  }
}
#endif

// Scalar fallbacks
MAT_INTERNAL_STATIC void
mat_gemm_unit_lower_t_scalar_(mat_elem_t *W, size_t ldw, const mat_elem_t *V,
                              size_t ldv, const mat_elem_t *R, size_t ldr,
                              size_t kb, size_t panel_rows, size_t N) {
  for (size_t ii = 0; ii < kb; ii++) {
    for (size_t jj = 0; jj < N; jj++) {
      mat_elem_t dot = R[ii * ldr + jj]; // diagonal 1
      for (size_t r = ii + 1; r < panel_rows; r++)
        dot += V[ii * ldv + r] * R[r * ldr + jj];
      W[ii * ldw + jj] = dot;
    }
  }
}

MAT_INTERNAL_STATIC void
mat_gemm_unit_lower_scalar_(mat_elem_t *C, size_t ldc, const mat_elem_t *V,
                            size_t ldv, const mat_elem_t *W, size_t ldw,
                            size_t panel_rows, size_t kb, size_t N) {
  for (size_t r = 0; r < panel_rows; r++) {
    size_t ii_max = (r < kb) ? r + 1 : kb;
    for (size_t jj = 0; jj < N; jj++) {
      mat_elem_t sum = 0;
      for (size_t ii = 0; ii < ii_max; ii++) {
        mat_elem_t v_val = (r == ii) ? 1.0f : V[ii * ldv + r];
        sum += v_val * W[ii * ldw + jj];
      }
      C[r * ldc + jj] -= sum;
    }
  }
}

MAT_INTERNAL_STATIC void
mat_gemm_q_unit_lower_scalar_(mat_elem_t *W, size_t ldw, const mat_elem_t *Q,
                              size_t ldq, const mat_elem_t *V, size_t ldv,
                              size_t M, size_t panel_rows, size_t kb) {
  for (size_t ii = 0; ii < M; ii++) {
    for (size_t jj = 0; jj < kb; jj++) {
      mat_elem_t dot = Q[ii * ldq + jj];
      for (size_t r = jj + 1; r < panel_rows; r++)
        dot += Q[ii * ldq + r] * V[jj * ldv + r];
      W[ii * ldw + jj] = dot;
    }
  }
}

MAT_INTERNAL_STATIC void mat_gemm_sub_w_unit_upper_scalar_(
    mat_elem_t *Q, size_t ldq, const mat_elem_t *W, size_t ldw,
    const mat_elem_t *V, size_t ldv, size_t M, size_t kb, size_t panel_rows) {
  for (size_t r = 0; r < panel_rows; r++) {
    size_t jj_max = (r < kb) ? r + 1 : kb;
    for (size_t ii = 0; ii < M; ii++) {
      mat_elem_t sum = 0;
      for (size_t jj = 0; jj < jj_max; jj++) {
        mat_elem_t v_val = (r == jj) ? 1.0f : V[jj * ldv + r];
        sum += W[ii * ldw + jj] * v_val;
      }
      Q[ii * ldq + r] -= sum;
    }
  }
}

// Column-major strided GEMM: C = alpha * A * B + beta * C
// Layout: A[i,k] = A[k*lda + i], B[k,j] = B[j*ldb + k], C[i,j] = C[j*ldc + i]
// Uses same strategy as optimized GEMM: transpose A, use 4x4 micro-kernel
// transA/transB: if MAT_TRANS, treat input as transposed without copying
#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void
mat_gemm_strided_colmajor_neon_(mat_elem_t *C, size_t ldc, mat_elem_t alpha,
                                const mat_elem_t *A, size_t lda,
                                mat_trans_t transA, const mat_elem_t *B,
                                size_t ldb, mat_trans_t transB, size_t M,
                                size_t K, size_t N, mat_elem_t beta) {
  // Scale C by beta first
  if (beta == 0) {
    for (size_t j = 0; j < N; j++)
      memset(&C[j * ldc], 0, M * sizeof(mat_elem_t));
  } else if (beta != 1) {
    for (size_t j = 0; j < N; j++) {
      mat_elem_t *Cj = &C[j * ldc];
      for (size_t i = 0; i < M; i++)
        Cj[i] *= beta;
    }
  }

  // Micro-kernel with K-blocking for cache locality
  // Both precisions use 8xN tiles with 16 accumulators for equal compute density
  // float32: 8x8 (2 vecs A × 2 vecs B × 4 lanes = 16 FMAs)
  // float64: 8x4 (4 vecs A × 2 vecs B × 2 lanes = 16 FMAs)
  const size_t KC = 256;
  const size_t W = MAT_NEON_WIDTH;        // 4 for f32, 2 for f64
  const size_t MR = 8;                    // Always 8 rows
  const size_t NR = 2 * W;                // 8 for f32, 4 for f64

  size_t M_MR = (M / MR) * MR;
  size_t N_NR = (N / NR) * NR;
  size_t npanels_a = M_MR / MR;

  // Allocate packing buffers for one K-block
  mat_elem_t *packed_a =
      (mat_elem_t *)mat_scratch_alloc_(npanels_a * KC * MR * sizeof(mat_elem_t));
  mat_elem_t *packed_b =
      (mat_elem_t *)mat_scratch_alloc_(KC * NR * sizeof(mat_elem_t));

  // K-blocking loop
  for (size_t k0 = 0; k0 < K; k0 += KC) {
    size_t kc = (k0 + KC <= K) ? KC : (K - k0);

    // Pack A panels for this K-block
    for (size_t p = 0; p < npanels_a; p++) {
      size_t i = p * MR;
      mat_elem_t *pa = packed_a + p * KC * MR;
      for (size_t kk = 0; kk < kc; kk++) {
        size_t k = k0 + kk;
        for (size_t ii = 0; ii < MR; ii++)
          // A[i,k]: normal = A[k*lda + i], transposed = A[i*lda + k]
          pa[kk * MR + ii] = transA ? A[(i + ii) * lda + k] : A[k * lda + i + ii];
      }
    }

    for (size_t j = 0; j < N_NR; j += NR) {
      // Pack B for this K-block
      for (size_t kk = 0; kk < kc; kk++) {
        size_t k = k0 + kk;
        for (size_t jj = 0; jj < NR; jj++)
          // B[k,j]: normal = B[j*ldb + k], transposed = B[k*ldb + j]
          packed_b[kk * NR + jj] =
              transB ? B[k * ldb + (j + jj)] : B[(j + jj) * ldb + k];
      }

      for (size_t p = 0; p < npanels_a; p++) {
        size_t i = p * MR;
        mat_elem_t *pa = packed_a + p * KC * MR;
        mat_elem_t *Cptr = &C[j * ldc + i];

        // 16 accumulators for 8xNR tile
        // float32: c[col][row_half] for col=0..7, row_half=0..1
        // float64: c[col][row_quarter] for col=0..3, row_quarter=0..3
        MAT_NEON_TYPE c00 = MAT_NEON_DUP(0), c01 = MAT_NEON_DUP(0);
        MAT_NEON_TYPE c02 = MAT_NEON_DUP(0), c03 = MAT_NEON_DUP(0);
        MAT_NEON_TYPE c04 = MAT_NEON_DUP(0), c05 = MAT_NEON_DUP(0);
        MAT_NEON_TYPE c06 = MAT_NEON_DUP(0), c07 = MAT_NEON_DUP(0);
        MAT_NEON_TYPE c10 = MAT_NEON_DUP(0), c11 = MAT_NEON_DUP(0);
        MAT_NEON_TYPE c12 = MAT_NEON_DUP(0), c13 = MAT_NEON_DUP(0);
        MAT_NEON_TYPE c14 = MAT_NEON_DUP(0), c15 = MAT_NEON_DUP(0);
        MAT_NEON_TYPE c16 = MAT_NEON_DUP(0), c17 = MAT_NEON_DUP(0);

        // Micro-kernel FMA macro for one k iteration
#if MAT_NEON_WIDTH == 4
        // float32: 2 A vectors (8 rows), 2 B vectors (8 cols), 16 FMAs
#define GEMM_FMA_ITER(pak, pbk)                                                \
  do {                                                                         \
    MAT_NEON_TYPE a0 = MAT_NEON_LOAD(pak);                                     \
    MAT_NEON_TYPE a1 = MAT_NEON_LOAD(pak + 4);                                 \
    MAT_NEON_TYPE b0 = MAT_NEON_LOAD(pbk);                                     \
    MAT_NEON_TYPE b1 = MAT_NEON_LOAD(pbk + 4);                                 \
    c00 = MAT_NEON_FMA_LANE(c00, a0, b0, 0);                                   \
    c01 = MAT_NEON_FMA_LANE(c01, a1, b0, 0);                                   \
    c02 = MAT_NEON_FMA_LANE(c02, a0, b0, 1);                                   \
    c03 = MAT_NEON_FMA_LANE(c03, a1, b0, 1);                                   \
    c04 = MAT_NEON_FMA_LANE(c04, a0, b0, 2);                                   \
    c05 = MAT_NEON_FMA_LANE(c05, a1, b0, 2);                                   \
    c06 = MAT_NEON_FMA_LANE(c06, a0, b0, 3);                                   \
    c07 = MAT_NEON_FMA_LANE(c07, a1, b0, 3);                                   \
    c10 = MAT_NEON_FMA_LANE(c10, a0, b1, 0);                                   \
    c11 = MAT_NEON_FMA_LANE(c11, a1, b1, 0);                                   \
    c12 = MAT_NEON_FMA_LANE(c12, a0, b1, 1);                                   \
    c13 = MAT_NEON_FMA_LANE(c13, a1, b1, 1);                                   \
    c14 = MAT_NEON_FMA_LANE(c14, a0, b1, 2);                                   \
    c15 = MAT_NEON_FMA_LANE(c15, a1, b1, 2);                                   \
    c16 = MAT_NEON_FMA_LANE(c16, a0, b1, 3);                                   \
    c17 = MAT_NEON_FMA_LANE(c17, a1, b1, 3);                                   \
  } while (0)
#else
        // float64: 4 A vectors (8 rows), 2 B vectors (4 cols), 16 FMAs
#define GEMM_FMA_ITER(pak, pbk)                                                \
  do {                                                                         \
    MAT_NEON_TYPE a0 = MAT_NEON_LOAD(pak);                                     \
    MAT_NEON_TYPE a1 = MAT_NEON_LOAD(pak + 2);                                 \
    MAT_NEON_TYPE a2 = MAT_NEON_LOAD(pak + 4);                                 \
    MAT_NEON_TYPE a3 = MAT_NEON_LOAD(pak + 6);                                 \
    MAT_NEON_TYPE b0 = MAT_NEON_LOAD(pbk);                                     \
    MAT_NEON_TYPE b1 = MAT_NEON_LOAD(pbk + 2);                                 \
    c00 = MAT_NEON_FMA_LANE(c00, a0, b0, 0);                                   \
    c01 = MAT_NEON_FMA_LANE(c01, a1, b0, 0);                                   \
    c02 = MAT_NEON_FMA_LANE(c02, a2, b0, 0);                                   \
    c03 = MAT_NEON_FMA_LANE(c03, a3, b0, 0);                                   \
    c04 = MAT_NEON_FMA_LANE(c04, a0, b0, 1);                                   \
    c05 = MAT_NEON_FMA_LANE(c05, a1, b0, 1);                                   \
    c06 = MAT_NEON_FMA_LANE(c06, a2, b0, 1);                                   \
    c07 = MAT_NEON_FMA_LANE(c07, a3, b0, 1);                                   \
    c10 = MAT_NEON_FMA_LANE(c10, a0, b1, 0);                                   \
    c11 = MAT_NEON_FMA_LANE(c11, a1, b1, 0);                                   \
    c12 = MAT_NEON_FMA_LANE(c12, a2, b1, 0);                                   \
    c13 = MAT_NEON_FMA_LANE(c13, a3, b1, 0);                                   \
    c14 = MAT_NEON_FMA_LANE(c14, a0, b1, 1);                                   \
    c15 = MAT_NEON_FMA_LANE(c15, a1, b1, 1);                                   \
    c16 = MAT_NEON_FMA_LANE(c16, a2, b1, 1);                                   \
    c17 = MAT_NEON_FMA_LANE(c17, a3, b1, 1);                                   \
  } while (0)
#endif

        // Unrolled by 4
        size_t kc4 = kc & ~(size_t)3;
        for (size_t kk = 0; kk < kc4; kk += 4) {
          mat_elem_t *pak = pa + kk * MR;
          mat_elem_t *pbk = packed_b + kk * NR;
          GEMM_FMA_ITER(pak, pbk);
          GEMM_FMA_ITER(pak + MR, pbk + NR);
          GEMM_FMA_ITER(pak + 2 * MR, pbk + 2 * NR);
          GEMM_FMA_ITER(pak + 3 * MR, pbk + 3 * NR);
        }
        // Remainder
        for (size_t kk = kc4; kk < kc; kk++) {
          GEMM_FMA_ITER(pa + kk * MR, packed_b + kk * NR);
        }
#undef GEMM_FMA_ITER

        // Store: C += alpha * partial_product
#if MAT_NEON_WIDTH == 4
        // float32: c[col][row_half], row_half offset = 4
#define GEMM_STORE_COL(col, c0, c1)                                            \
  MAT_NEON_STORE(Cptr + (col)*ldc,                                             \
                 MAT_NEON_ADD(MAT_NEON_LOAD(Cptr + (col)*ldc), c0));            \
  MAT_NEON_STORE(Cptr + 4 + (col)*ldc,                                         \
                 MAT_NEON_ADD(MAT_NEON_LOAD(Cptr + 4 + (col)*ldc), c1))
#define GEMM_STORE_COL_ALPHA(col, c0, c1)                                      \
  MAT_NEON_STORE(Cptr + (col)*ldc, MAT_NEON_ADD(MAT_NEON_LOAD(Cptr + (col)*ldc), MAT_NEON_MUL(av, c0))); \
  MAT_NEON_STORE(Cptr + 4 + (col)*ldc, MAT_NEON_ADD(MAT_NEON_LOAD(Cptr + 4 + (col)*ldc), MAT_NEON_MUL(av, c1)))
#else
        // float64: c[col][row_quarter], row_quarter offset = 2
#define GEMM_STORE_COL(col, c0, c1, c2, c3)                                    \
  MAT_NEON_STORE(Cptr + (col)*ldc,                                             \
                 MAT_NEON_ADD(MAT_NEON_LOAD(Cptr + (col)*ldc), c0));            \
  MAT_NEON_STORE(Cptr + 2 + (col)*ldc,                                         \
                 MAT_NEON_ADD(MAT_NEON_LOAD(Cptr + 2 + (col)*ldc), c1));        \
  MAT_NEON_STORE(Cptr + 4 + (col)*ldc,                                         \
                 MAT_NEON_ADD(MAT_NEON_LOAD(Cptr + 4 + (col)*ldc), c2));        \
  MAT_NEON_STORE(Cptr + 6 + (col)*ldc,                                         \
                 MAT_NEON_ADD(MAT_NEON_LOAD(Cptr + 6 + (col)*ldc), c3))
#define GEMM_STORE_COL_ALPHA(col, c0, c1, c2, c3)                              \
  MAT_NEON_STORE(Cptr + (col)*ldc, MAT_NEON_ADD(MAT_NEON_LOAD(Cptr + (col)*ldc), MAT_NEON_MUL(av, c0))); \
  MAT_NEON_STORE(Cptr + 2 + (col)*ldc, MAT_NEON_ADD(MAT_NEON_LOAD(Cptr + 2 + (col)*ldc), MAT_NEON_MUL(av, c1))); \
  MAT_NEON_STORE(Cptr + 4 + (col)*ldc, MAT_NEON_ADD(MAT_NEON_LOAD(Cptr + 4 + (col)*ldc), MAT_NEON_MUL(av, c2))); \
  MAT_NEON_STORE(Cptr + 6 + (col)*ldc, MAT_NEON_ADD(MAT_NEON_LOAD(Cptr + 6 + (col)*ldc), MAT_NEON_MUL(av, c3)))
#endif

        if (alpha == 1) {
#if MAT_NEON_WIDTH == 4
          GEMM_STORE_COL(0, c00, c01);
          GEMM_STORE_COL(1, c02, c03);
          GEMM_STORE_COL(2, c04, c05);
          GEMM_STORE_COL(3, c06, c07);
          GEMM_STORE_COL(4, c10, c11);
          GEMM_STORE_COL(5, c12, c13);
          GEMM_STORE_COL(6, c14, c15);
          GEMM_STORE_COL(7, c16, c17);
#else
          GEMM_STORE_COL(0, c00, c01, c02, c03);
          GEMM_STORE_COL(1, c04, c05, c06, c07);
          GEMM_STORE_COL(2, c10, c11, c12, c13);
          GEMM_STORE_COL(3, c14, c15, c16, c17);
#endif
        } else {
          MAT_NEON_TYPE av = MAT_NEON_DUP(alpha);
#if MAT_NEON_WIDTH == 4
          GEMM_STORE_COL_ALPHA(0, c00, c01);
          GEMM_STORE_COL_ALPHA(1, c02, c03);
          GEMM_STORE_COL_ALPHA(2, c04, c05);
          GEMM_STORE_COL_ALPHA(3, c06, c07);
          GEMM_STORE_COL_ALPHA(4, c10, c11);
          GEMM_STORE_COL_ALPHA(5, c12, c13);
          GEMM_STORE_COL_ALPHA(6, c14, c15);
          GEMM_STORE_COL_ALPHA(7, c16, c17);
#else
          GEMM_STORE_COL_ALPHA(0, c00, c01, c02, c03);
          GEMM_STORE_COL_ALPHA(1, c04, c05, c06, c07);
          GEMM_STORE_COL_ALPHA(2, c10, c11, c12, c13);
          GEMM_STORE_COL_ALPHA(3, c14, c15, c16, c17);
#endif
        }
#undef GEMM_STORE_COL
#undef GEMM_STORE_COL_ALPHA
      }

      // Remainder i rows for this j panel (scalar)
      for (size_t i = M_MR; i < M; i++) {
        for (size_t jj = 0; jj < NR; jj++) {
          mat_elem_t sum = 0;
          for (size_t kk = 0; kk < kc; kk++) {
            size_t k = k0 + kk;
            mat_elem_t a_ik = transA ? A[i * lda + k] : A[k * lda + i];
            sum += a_ik * packed_b[kk * NR + jj];
          }
          C[(j + jj) * ldc + i] += alpha * sum;
        }
      }
    }

    // Remainder j columns (scalar)
    for (size_t jj = N_NR; jj < N; jj++) {
      for (size_t i = 0; i < M; i++) {
        mat_elem_t sum = 0;
        for (size_t kk = 0; kk < kc; kk++) {
          size_t k = k0 + kk;
          mat_elem_t a_ik = transA ? A[i * lda + k] : A[k * lda + i];
          mat_elem_t b_kj = transB ? B[k * ldb + jj] : B[jj * ldb + k];
          sum += a_ik * b_kj;
        }
        C[jj * ldc + i] += alpha * sum;
      }
    }
  }

#ifndef MAT_NO_SCRATCH
  mat_scratch_reset_();
#else
  mat_scratch_free_(packed_b);
  mat_scratch_free_(packed_a);
#endif
}
#endif // MAT_HAS_ARM_NEON

MAT_INTERNAL_STATIC void
mat_gemm_strided_colmajor_(mat_elem_t *C, size_t ldc, mat_elem_t alpha,
                           const mat_elem_t *A, size_t lda, mat_trans_t transA,
                           const mat_elem_t *B, size_t ldb, mat_trans_t transB,
                           size_t M, size_t K, size_t N, mat_elem_t beta) {
#ifdef MAT_HAS_ARM_NEON
  mat_gemm_strided_colmajor_neon_(C, ldc, alpha, A, lda, transA, B, ldb, transB,
                                  M, K, N, beta);
#else
  // Scalar fallback
  if (beta == 0) {
    for (size_t j = 0; j < N; j++)
      memset(&C[j * ldc], 0, M * sizeof(mat_elem_t));
  } else if (beta != 1) {
    for (size_t j = 0; j < N; j++)
      for (size_t i = 0; i < M; i++)
        C[j * ldc + i] *= beta;
  }
  // C[:,j] += alpha * sum_k op(A)[:,k] * op(B)[k,j]
  for (size_t j = 0; j < N; j++) {
    for (size_t k = 0; k < K; k++) {
      mat_elem_t b_kj = transB ? B[k * ldb + j] : B[j * ldb + k];
      mat_elem_t ab = alpha * b_kj;
      for (size_t i = 0; i < M; i++) {
        mat_elem_t a_ik = transA ? A[i * lda + k] : A[k * lda + i];
        C[j * ldc + i] += a_ik * ab;
      }
    }
  }
#endif
}

// C = alpha * A * B + beta * C (BLAS Level-3: gemm)
#ifdef MAT_HAS_ARM_NEON
// Forward declaration for fallback
MAT_INTERNAL_STATIC void mat_gemm_scalar_(Mat *C, mat_elem_t alpha,
                                          const Mat *A, const Mat *B,
                                          mat_elem_t beta);

MAT_INTERNAL_STATIC void mat_gemm_neon_(Mat *C, mat_elem_t alpha, const Mat *A,
                                        const Mat *B, mat_elem_t beta) {
  // Column-major: delegate to strided implementation
  // For column-major Mat, leading dimension = rows
  mat_gemm_strided_colmajor_neon_(C->data, C->rows, alpha, A->data, A->rows,
                                  MAT_NO_TRANS, B->data, B->rows, MAT_NO_TRANS,
                                  A->rows, A->cols, B->cols, beta);
}
#endif // MAT_HAS_ARM_NEON

MAT_INTERNAL_STATIC void mat_gemm_scalar_(Mat *C, mat_elem_t alpha,
                                          const Mat *A, const Mat *B,
                                          mat_elem_t beta) {
  size_t M = A->rows;
  size_t K = A->cols;
  size_t N = B->cols;

  // Scale C by beta first, then accumulate
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      MAT_SET(C, i, j, MAT_AT(C, i, j) * beta);
    }
  }

  // ikj loop order for cache-friendly access (row-major)
  // For column-major, jki would be better, but this still works correctly
#ifdef MAT_HAS_OPENMP
#pragma omp parallel for schedule(static) if (M * N * K >= MAT_OMP_THRESHOLD)
#endif
  for (size_t i = 0; i < M; i++) {
    for (size_t k = 0; k < K; k++) {
      mat_elem_t aik = alpha * MAT_AT(A, i, k);
      for (size_t j = 0; j < N; j++) {
        MAT_SET(C, i, j, MAT_AT(C, i, j) + aik * MAT_AT(B, k, j));
      }
    }
  }
}

MATDEF void mat_gemm(Mat *C, mat_elem_t alpha, const Mat *A, const Mat *B,
                     mat_elem_t beta) {
  MAT_ASSERT_MAT(C);
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(B);
  MAT_ASSERT(A->cols == B->rows);
  MAT_ASSERT(C->rows == A->rows && C->cols == B->cols);

#ifdef MAT_HAS_ARM_NEON
  mat_gemm_neon_(C, alpha, A, B, beta);
#else
  mat_gemm_scalar_(C, alpha, A, B, beta);
#endif
}

/* Structure Operations */

#define MAT_T_BLOCK 32

#ifdef MAT_HAS_ARM_NEON
// Forward declaration for fallback
MAT_INTERNAL_STATIC void mat_t_scalar_(Mat *out, const Mat *m);

MAT_INTERNAL_STATIC void mat_t_neon_(Mat *out, const Mat *m) {
  // Column-major NEON transpose
  // Input: m is rows×cols, m[i,j] = src[j*rows + i] (column j is contiguous)
  // Output: out is cols×rows, out[j,i] = dst[i*cols + j] (column i is contiguous)
  // We want: out[j,i] = m[i,j], so dst[i*cols + j] = src[j*rows + i]
  size_t rows = m->rows;
  size_t cols = m->cols;
  const mat_elem_t *src = m->data;
  mat_elem_t *dst = out->data;

  size_t full_rows = (rows / MAT_T_BLOCK) * MAT_T_BLOCK;
  size_t full_cols = (cols / MAT_T_BLOCK) * MAT_T_BLOCK;

#if defined(MAT_HAS_OPENMP)
#pragma omp parallel for collapse(2)                                           \
    schedule(static) if (rows * cols >= MAT_OMP_THRESHOLD)
#endif
  for (size_t ii = 0; ii < full_rows; ii += MAT_T_BLOCK) {
    for (size_t jj = 0; jj < full_cols; jj += MAT_T_BLOCK) {
#ifdef MAT_DOUBLE_PRECISION
      // float64x2_t: process 2x2 blocks
      // Load from columns jj, jj+1 of input (each column is contiguous)
      // Store to columns ii, ii+1 of output (each column is contiguous)
      for (size_t i = ii; i < ii + MAT_T_BLOCK; i += 2) {
        for (size_t j = jj; j < jj + MAT_T_BLOCK; j += 2) {
          // Load 2 elements from column j and j+1, rows i and i+1
          float64x2_t c0 = vld1q_f64(&src[(j + 0) * rows + i]);  // col j
          float64x2_t c1 = vld1q_f64(&src[(j + 1) * rows + i]);  // col j+1
          // Transpose 2x2
          float64x2_t t0 = vzip1q_f64(c0, c1);
          float64x2_t t1 = vzip2q_f64(c0, c1);
          // Store to column i and i+1 of output
          vst1q_f64(&dst[(i + 0) * cols + j], t0);
          vst1q_f64(&dst[(i + 1) * cols + j], t1);
        }
      }
#else
      // float32x4_t: process 4x4 blocks
      // Load from columns jj..jj+3 of input (each column is contiguous)
      // Store to columns ii..ii+3 of output (each column is contiguous)
      for (size_t i = ii; i < ii + MAT_T_BLOCK; i += 4) {
        for (size_t j = jj; j < jj + MAT_T_BLOCK; j += 4) {
          // Load 4 elements from each of 4 columns
          float32x4_t c0 = vld1q_f32(&src[(j + 0) * rows + i]);
          float32x4_t c1 = vld1q_f32(&src[(j + 1) * rows + i]);
          float32x4_t c2 = vld1q_f32(&src[(j + 2) * rows + i]);
          float32x4_t c3 = vld1q_f32(&src[(j + 3) * rows + i]);
          // Transpose 4x4 using NEON intrinsics
          float32x4x2_t p01 = vtrnq_f32(c0, c1);
          float32x4x2_t p23 = vtrnq_f32(c2, c3);
          float32x4_t t0 =
              vcombine_f32(vget_low_f32(p01.val[0]), vget_low_f32(p23.val[0]));
          float32x4_t t1 =
              vcombine_f32(vget_low_f32(p01.val[1]), vget_low_f32(p23.val[1]));
          float32x4_t t2 =
              vcombine_f32(vget_high_f32(p01.val[0]), vget_high_f32(p23.val[0]));
          float32x4_t t3 =
              vcombine_f32(vget_high_f32(p01.val[1]), vget_high_f32(p23.val[1]));
          // Store to 4 columns of output
          vst1q_f32(&dst[(i + 0) * cols + j], t0);
          vst1q_f32(&dst[(i + 1) * cols + j], t1);
          vst1q_f32(&dst[(i + 2) * cols + j], t2);
          vst1q_f32(&dst[(i + 3) * cols + j], t3);
        }
      }
#endif
    }
  }

  // Edge cases: right edge (cols not multiple of block)
  for (size_t ii = 0; ii < full_rows; ii += MAT_T_BLOCK) {
    for (size_t i = ii; i < ii + MAT_T_BLOCK; i++) {
      for (size_t j = full_cols; j < cols; j++) {
        dst[i * cols + j] = src[j * rows + i];
      }
    }
  }

  // Edge cases: bottom edge (rows not multiple of block)
  for (size_t i = full_rows; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      dst[i * cols + j] = src[j * rows + i];
    }
  }
}
#endif

MAT_INTERNAL_STATIC void mat_t_scalar_(Mat *out, const Mat *m) {
  size_t rows = m->rows;
  size_t cols = m->cols;

#ifdef MAT_HAS_OPENMP
#pragma omp parallel for collapse(2)                                           \
    schedule(static) if (rows * cols >= MAT_OMP_THRESHOLD)
#endif
  for (size_t ii = 0; ii < rows; ii += MAT_T_BLOCK) {
    for (size_t jj = 0; jj < cols; jj += MAT_T_BLOCK) {
      size_t i_end = (ii + MAT_T_BLOCK < rows) ? ii + MAT_T_BLOCK : rows;
      size_t j_end = (jj + MAT_T_BLOCK < cols) ? jj + MAT_T_BLOCK : cols;
      for (size_t i = ii; i < i_end; i++) {
        for (size_t j = jj; j < j_end; j++) {
          // out[j,i] = m[i,j]
          MAT_SET(out, j, i, MAT_AT(m, i, j));
        }
      }
    }
  }
}

MATDEF void mat_t(Mat *out, const Mat *m) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(m);
  MAT_ASSERT(out->rows == m->cols && out->cols == m->rows);

#ifdef MAT_HAS_ARM_NEON
  mat_t_neon_(out, m);
#else
  mat_t_scalar_(out, m);
#endif
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

MATDEF Mat *mat_slice(const Mat *m, size_t row_start, size_t row_end,
                      size_t col_start, size_t col_end) {
  MAT_ASSERT_MAT(m);
  MAT_ASSERT(row_start < row_end && row_end <= m->rows);
  MAT_ASSERT(col_start < col_end && col_end <= m->cols);

  size_t out_rows = row_end - row_start;
  size_t out_cols = col_end - col_start;
  Mat *result = mat_mat(out_rows, out_cols);

// Column-major: columns are contiguous, copy column by column
  for (size_t j = 0; j < out_cols; j++) {
    memcpy(&result->data[j * out_rows],
           &m->data[(col_start + j) * m->rows + row_start],
           out_rows * sizeof(mat_elem_t));
  }

  return result;
}

MATDEF void mat_slice_set(Mat *m, size_t row_start, size_t col_start,
                          const Mat *src) {
  MAT_ASSERT_MAT(m);
  MAT_ASSERT_MAT(src);
  MAT_ASSERT(row_start + src->rows <= m->rows);
  MAT_ASSERT(col_start + src->cols <= m->cols);

// Column-major: columns are contiguous, copy column by column
  for (size_t j = 0; j < src->cols; j++) {
    memcpy(&m->data[(col_start + j) * m->rows + row_start],
           &src->data[j * src->rows], src->rows * sizeof(mat_elem_t));
  }
}

MATDEF void mat_hcat(Mat *out, const Mat *a, const Mat *b) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);
  MAT_ASSERT_MAT(b);
  MAT_ASSERT(a->rows == b->rows && "mat_hcat: row count must match");
  MAT_ASSERT(out->rows == a->rows && out->cols == a->cols + b->cols);

// Column-major: [a|b] = copy columns of a, then columns of b
  memcpy(out->data, a->data, a->rows * a->cols * sizeof(mat_elem_t));
  memcpy(&out->data[a->rows * a->cols], b->data,
         b->rows * b->cols * sizeof(mat_elem_t));
}

MATDEF void mat_vcat(Mat *out, const Mat *a, const Mat *b) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);
  MAT_ASSERT_MAT(b);
  MAT_ASSERT(a->cols == b->cols && "mat_vcat: column count must match");
  MAT_ASSERT(out->rows == a->rows + b->rows && out->cols == a->cols);

// Column-major: [a; b] = copy column by column, a then b for each column
  for (size_t j = 0; j < a->cols; j++) {
    memcpy(&out->data[j * out->rows], &a->data[j * a->rows],
           a->rows * sizeof(mat_elem_t));
    memcpy(&out->data[j * out->rows + a->rows], &b->data[j * b->rows],
           b->rows * sizeof(mat_elem_t));
  }
}

MATDEF Vec *mat_row(const Mat *m, size_t row) {
  MAT_ASSERT_MAT(m);
  MAT_ASSERT(row < m->rows && "mat_row: row index out of bounds");

  Vec *v = mat_vec(m->cols);
  // Column-major: rows are strided
  for (size_t j = 0; j < m->cols; j++) {
    v->data[j] = MAT_AT(m, row, j);
  }
  return v;
}

MATDEF Vec *mat_col(const Mat *m, size_t col) {
  MAT_ASSERT_MAT(m);
  MAT_ASSERT(col < m->cols && "mat_col: column index out of bounds");

  Vec *v = mat_vec(m->rows);
  // Column-major: columns are contiguous
  memcpy(v->data, &m->data[col * m->rows], m->rows * sizeof(mat_elem_t));
  return v;
}

MATDEF Vec mat_row_view(const Mat *m, size_t row) {
  MAT_ASSERT_MAT(m);
  MAT_ASSERT(row < m->rows && "mat_row_view: row index out of bounds");

// Column-major: rows are strided, cannot provide a contiguous view
  // This is a limitation - consider using mat_row() instead
  MAT_ASSERT(0 && "mat_row_view not supported in column-major mode");
  Vec v = {0};
  return v;
}

/* Diagonal Operations */

MATDEF Vec *mat_diag(const Mat *m) {
  MAT_ASSERT_SQUARE(m);

  Vec *d = mat_vec(m->rows);

  for (size_t i = 0; i < d->rows; i++) {
    d->data[i] = MAT_AT(m, i, i);
  }

  return d;
}

MATDEF Mat *mat_diag_from(size_t dim, const mat_elem_t *values) {
  MAT_ASSERT(values != NULL);
  MAT_ASSERT(dim > 0);

  Mat *result = mat_mat(dim, dim);

  for (size_t i = 0; i < dim; i++) {
    MAT_SET(result, i, i, values[i]);
  }

  return result;
}

/* Reduction Operations */

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC mat_elem_t mat_sum_neon_(const Mat *a) {
  size_t len = a->rows * a->cols;
  mat_elem_t *pa = a->data;

  MAT_NEON_TYPE vsum0 = MAT_NEON_DUP(0);
  MAT_NEON_TYPE vsum1 = MAT_NEON_DUP(0);
  MAT_NEON_TYPE vsum2 = MAT_NEON_DUP(0);
  MAT_NEON_TYPE vsum3 = MAT_NEON_DUP(0);

  size_t i = 0;
  for (; i + MAT_NEON_WIDTH * 4 <= len; i += MAT_NEON_WIDTH * 4) {
    vsum0 = MAT_NEON_ADD(vsum0, MAT_NEON_LOAD(&pa[i]));
    vsum1 = MAT_NEON_ADD(vsum1, MAT_NEON_LOAD(&pa[i + MAT_NEON_WIDTH]));
    vsum2 = MAT_NEON_ADD(vsum2, MAT_NEON_LOAD(&pa[i + MAT_NEON_WIDTH * 2]));
    vsum3 = MAT_NEON_ADD(vsum3, MAT_NEON_LOAD(&pa[i + MAT_NEON_WIDTH * 3]));
  }

  vsum0 = MAT_NEON_ADD(vsum0, vsum1);
  vsum2 = MAT_NEON_ADD(vsum2, vsum3);
  vsum0 = MAT_NEON_ADD(vsum0, vsum2);
  mat_elem_t sum = MAT_NEON_ADDV(vsum0);

  for (; i < len; i++) {
    sum += pa[i];
  }

  return sum;
}
#endif

MAT_INTERNAL_STATIC mat_elem_t mat_sum_scalar_(const Mat *a) {
  size_t len = a->rows * a->cols;
  mat_elem_t *pa = a->data;

  mat_elem_t sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;

  size_t i = 0;
  for (; i + 4 <= len; i += 4) {
    sum0 += pa[i];
    sum1 += pa[i + 1];
    sum2 += pa[i + 2];
    sum3 += pa[i + 3];
  }

  mat_elem_t sum = sum0 + sum1 + sum2 + sum3;

  for (; i < len; i++) {
    sum += pa[i];
  }

  return sum;
}

MATDEF mat_elem_t mat_sum(const Mat *a) {
  MAT_ASSERT_MAT(a);

#ifdef MAT_HAS_ARM_NEON
  return mat_sum_neon_(a);
#else
  return mat_sum_scalar_(a);
#endif
}

MATDEF mat_elem_t mat_mean(const Mat *a) {
  MAT_ASSERT_MAT(a);
  return mat_sum(a) / (mat_elem_t)(a->rows * a->cols);
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC mat_elem_t mat_min_neon_(const Mat *a) {
  size_t len = a->rows * a->cols;
  mat_elem_t *pa = a->data;

  MAT_NEON_TYPE vmin0 = MAT_NEON_LOAD(&pa[0]);
  MAT_NEON_TYPE vmin1 = vmin0;
  MAT_NEON_TYPE vmin2 = vmin0;
  MAT_NEON_TYPE vmin3 = vmin0;

  size_t i = MAT_NEON_WIDTH;
  for (; i + MAT_NEON_WIDTH * 4 <= len; i += MAT_NEON_WIDTH * 4) {
    MAT_NEON_TYPE va0 = MAT_NEON_LOAD(&pa[i]);
    MAT_NEON_TYPE va1 = MAT_NEON_LOAD(&pa[i + MAT_NEON_WIDTH]);
    MAT_NEON_TYPE va2 = MAT_NEON_LOAD(&pa[i + MAT_NEON_WIDTH * 2]);
    MAT_NEON_TYPE va3 = MAT_NEON_LOAD(&pa[i + MAT_NEON_WIDTH * 3]);

#ifdef MAT_DOUBLE_PRECISION
    vmin0 = vminq_f64(vmin0, va0);
    vmin1 = vminq_f64(vmin1, va1);
    vmin2 = vminq_f64(vmin2, va2);
    vmin3 = vminq_f64(vmin3, va3);
#else
    vmin0 = vminq_f32(vmin0, va0);
    vmin1 = vminq_f32(vmin1, va1);
    vmin2 = vminq_f32(vmin2, va2);
    vmin3 = vminq_f32(vmin3, va3);
#endif
  }

#ifdef MAT_DOUBLE_PRECISION
  vmin0 = vminq_f64(vmin0, vmin1);
  vmin2 = vminq_f64(vmin2, vmin3);
  vmin0 = vminq_f64(vmin0, vmin2);
  mat_elem_t min_val = vminvq_f64(vmin0);
#else
  vmin0 = vminq_f32(vmin0, vmin1);
  vmin2 = vminq_f32(vmin2, vmin3);
  vmin0 = vminq_f32(vmin0, vmin2);
  mat_elem_t min_val = vminvq_f32(vmin0);
#endif

  for (; i < len; i++) {
    if (pa[i] < min_val)
      min_val = pa[i];
  }

  return min_val;
}
#endif

MAT_INTERNAL_STATIC mat_elem_t mat_min_scalar_(const Mat *a) {
  size_t len = a->rows * a->cols;
  mat_elem_t *pa = a->data;

  mat_elem_t min_val = pa[0];

  for (size_t i = 1; i < len; i++) {
    if (pa[i] < min_val)
      min_val = pa[i];
  }

  return min_val;
}

MATDEF mat_elem_t mat_min(const Mat *a) {
  MAT_ASSERT_MAT(a);

#ifdef MAT_HAS_ARM_NEON
  return mat_min_neon_(a);
#else
  return mat_min_scalar_(a);
#endif
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC mat_elem_t mat_max_neon_(const Mat *a) {
  size_t len = a->rows * a->cols;
  mat_elem_t *pa = a->data;

  MAT_NEON_TYPE vmax0 = MAT_NEON_LOAD(&pa[0]);
  MAT_NEON_TYPE vmax1 = vmax0;
  MAT_NEON_TYPE vmax2 = vmax0;
  MAT_NEON_TYPE vmax3 = vmax0;

  size_t i = MAT_NEON_WIDTH;
  for (; i + MAT_NEON_WIDTH * 4 <= len; i += MAT_NEON_WIDTH * 4) {
    MAT_NEON_TYPE va0 = MAT_NEON_LOAD(&pa[i]);
    MAT_NEON_TYPE va1 = MAT_NEON_LOAD(&pa[i + MAT_NEON_WIDTH]);
    MAT_NEON_TYPE va2 = MAT_NEON_LOAD(&pa[i + MAT_NEON_WIDTH * 2]);
    MAT_NEON_TYPE va3 = MAT_NEON_LOAD(&pa[i + MAT_NEON_WIDTH * 3]);

    vmax0 = MAT_NEON_MAX(vmax0, va0);
    vmax1 = MAT_NEON_MAX(vmax1, va1);
    vmax2 = MAT_NEON_MAX(vmax2, va2);
    vmax3 = MAT_NEON_MAX(vmax3, va3);
  }

  vmax0 = MAT_NEON_MAX(vmax0, vmax1);
  vmax2 = MAT_NEON_MAX(vmax2, vmax3);
  vmax0 = MAT_NEON_MAX(vmax0, vmax2);
  mat_elem_t max_val = MAT_NEON_MAXV(vmax0);

  for (; i < len; i++) {
    if (pa[i] > max_val)
      max_val = pa[i];
  }

  return max_val;
}
#endif

MAT_INTERNAL_STATIC mat_elem_t mat_max_scalar_(const Mat *a) {
  size_t len = a->rows * a->cols;
  mat_elem_t *pa = a->data;

  mat_elem_t max_val = pa[0];

  for (size_t i = 1; i < len; i++) {
    if (pa[i] > max_val)
      max_val = pa[i];
  }

  return max_val;
}

MATDEF mat_elem_t mat_max(const Mat *a) {
  MAT_ASSERT_MAT(a);

#ifdef MAT_HAS_ARM_NEON
  return mat_max_neon_(a);
#else
  return mat_max_scalar_(a);
#endif
}

MATDEF void mat_sum_axis(Vec *out, const Mat *a, int axis) {
  MAT_ASSERT_MAT(a);
  MAT_ASSERT_MAT(out);

  if (axis == 0) {
    // Sum along columns: result has shape (rows, 1) = (a->rows,)
    MAT_ASSERT(out->rows * out->cols == a->rows &&
               "mat_sum_axis: output size must match rows");

    for (size_t i = 0; i < a->rows; i++) {
      mat_elem_t sum = 0;
      for (size_t j = 0; j < a->cols; j++) {
        sum += MAT_AT(a, i, j);
      }
      out->data[i] = sum;
    }
  } else {
    // Sum along rows (axis=1): result has shape (1, cols) = (a->cols,)
    MAT_ASSERT(out->rows * out->cols == a->cols &&
               "mat_sum_axis: output size must match cols");

    // Zero output
    for (size_t j = 0; j < a->cols; j++) {
      out->data[j] = 0;
    }

    for (size_t i = 0; i < a->rows; i++) {
      for (size_t j = 0; j < a->cols; j++) {
        out->data[j] += MAT_AT(a, i, j);
      }
    }
  }
}

MATDEF size_t mat_argmin(const Mat *a) {
  MAT_ASSERT_MAT(a);

  size_t len = a->rows * a->cols;
  mat_elem_t *pa = a->data;

  mat_elem_t min_val = pa[0];
  size_t min_idx = 0;

  for (size_t i = 1; i < len; i++) {
    if (pa[i] < min_val) {
      min_val = pa[i];
      min_idx = i;
    }
  }

  return min_idx;
}

MATDEF size_t mat_argmax(const Mat *a) {
  MAT_ASSERT_MAT(a);

  size_t len = a->rows * a->cols;
  mat_elem_t *pa = a->data;

  mat_elem_t max_val = pa[0];
  size_t max_idx = 0;

  for (size_t i = 1; i < len; i++) {
    if (pa[i] > max_val) {
      max_val = pa[i];
      max_idx = i;
    }
  }

  return max_idx;
}

MATDEF mat_elem_t mat_std(const Mat *a) {
  MAT_ASSERT_MAT(a);

  size_t n = a->rows * a->cols;
  mat_elem_t mean = mat_mean(a);

  mat_elem_t *pa = a->data;
  mat_elem_t sum_sq = 0;

  for (size_t i = 0; i < n; i++) {
    mat_elem_t diff = pa[i] - mean;
    sum_sq += diff * diff;
  }

#ifdef MAT_DOUBLE_PRECISION
  return sqrt(sum_sq / (mat_elem_t)n);
#else
  return sqrtf(sum_sq / (mat_elem_t)n);
#endif
}

/* Norms */

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

MATDEF mat_elem_t mat_norm2(const Mat *a) { return mat_norm_fro(a); }

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC mat_elem_t mat_norm_max_neon_(const Mat *a) {
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
    MAT_NEON_TYPE va2 =
        MAT_NEON_ABS(MAT_NEON_LOAD(&pa[i + MAT_NEON_WIDTH * 2]));
    MAT_NEON_TYPE va3 =
        MAT_NEON_ABS(MAT_NEON_LOAD(&pa[i + MAT_NEON_WIDTH * 3]));

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
    if (v > max)
      max = v;
  }

  return max;
}
#endif

MAT_INTERNAL_STATIC mat_elem_t mat_norm_max_scalar_(const Mat *a) {
  size_t len = a->rows * a->cols;
  mat_elem_t max = fabs(a->data[0]);
  for (size_t i = 1; i < len; i++) {
    mat_elem_t v = fabs(a->data[i]);
    if (v > max)
      max = v;
  }
  return max;
}

MATDEF mat_elem_t mat_norm_max(const Mat *a) {
  MAT_ASSERT_MAT(a);

#ifdef MAT_HAS_ARM_NEON
  return mat_norm_max_neon_(a);
#else
  return mat_norm_max_scalar_(a);
#endif
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC mat_elem_t mat_norm_fro_neon_(const Mat *a) {
  size_t len = a->rows * a->cols;
  mat_elem_t *pa = a->data;

  MAT_ACC_TYPE vsum0 = MAT_ACC_ZERO;
  MAT_ACC_TYPE vsum1 = MAT_ACC_ZERO;
  MAT_ACC_TYPE vsum2 = MAT_ACC_ZERO;
  MAT_ACC_TYPE vsum3 = MAT_ACC_ZERO;

  size_t i = 0;
  for (; i + MAT_ACC_WIDTH * 4 <= len; i += MAT_ACC_WIDTH * 4) {
    MAT_ACC_LOAD_SQ(vsum0, &pa[i]);
    MAT_ACC_LOAD_SQ(vsum1, &pa[i + MAT_ACC_WIDTH]);
    MAT_ACC_LOAD_SQ(vsum2, &pa[i + MAT_ACC_WIDTH * 2]);
    MAT_ACC_LOAD_SQ(vsum3, &pa[i + MAT_ACC_WIDTH * 3]);
  }

  for (; i + MAT_ACC_WIDTH <= len; i += MAT_ACC_WIDTH) {
    MAT_ACC_LOAD_SQ(vsum0, &pa[i]);
  }

  vsum0 = MAT_ACC_ADD(vsum0, vsum1);
  vsum2 = MAT_ACC_ADD(vsum2, vsum3);
  vsum0 = MAT_ACC_ADD(vsum0, vsum2);
  double sum = MAT_ACC_ADDV(vsum0);

  for (; i < len; i++) {
    double v = pa[i];
    sum += v * v;
  }

  return MAT_SQRT(sum);
}
#endif

MAT_INTERNAL_STATIC mat_elem_t mat_norm_fro_scalar_(const Mat *a) {
  size_t len = a->rows * a->cols;
  mat_elem_t sum = 0;
  for (size_t i = 0; i < len; i++) {
    sum += a->data[i] * a->data[i];
  }
  return MAT_SQRT(sum);
}

MATDEF mat_elem_t mat_norm_fro(const Mat *a) {
  MAT_ASSERT_MAT(a);

#ifdef MAT_HAS_ARM_NEON
  return mat_norm_fro_neon_(a);
#else
  return mat_norm_fro_scalar_(a);
#endif
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC mat_elem_t mat_norm_fro_fast_neon_(const Mat *a) {
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

  return MAT_SQRT(sum);
}
#endif

MATDEF mat_elem_t mat_norm_fro_fast(const Mat *a) {
  MAT_ASSERT_MAT(a);

#ifdef MAT_HAS_ARM_NEON
  return mat_norm_fro_fast_neon_(a);
#else
  return mat_norm_fro_scalar_(a);
#endif
}

/* Matrix Properties */

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
MAT_INTERNAL_STATIC mat_elem_t mat_nnz_neon_(const Mat *a) {
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

    MAT_NEON_UTYPE mask0 =
        MAT_NEON_AND_U(MAT_NEON_MVN_U(MAT_NEON_CEQ(va0, vzero)), vone);
    MAT_NEON_UTYPE mask1 =
        MAT_NEON_AND_U(MAT_NEON_MVN_U(MAT_NEON_CEQ(va1, vzero)), vone);
    MAT_NEON_UTYPE mask2 =
        MAT_NEON_AND_U(MAT_NEON_MVN_U(MAT_NEON_CEQ(va2, vzero)), vone);
    MAT_NEON_UTYPE mask3 =
        MAT_NEON_AND_U(MAT_NEON_MVN_U(MAT_NEON_CEQ(va3, vzero)), vone);

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
    if (pa[i] != 0)
      count++;
  }

  return (mat_elem_t)count;
}
#endif

MAT_INTERNAL_STATIC mat_elem_t mat_nnz_scalar_(const Mat *a) {
  size_t len = a->rows * a->cols;
  mat_elem_t count = 0;
  for (size_t i = 0; i < len; i++) {
    if (a->data[i] != 0)
      count++;
  }
  return count;
}

MATDEF mat_elem_t mat_nnz(const Mat *a) {
  MAT_ASSERT_MAT(a);

#ifdef MAT_HAS_ARM_NEON
  return mat_nnz_neon_(a);
#else
  return mat_nnz_scalar_(a);
#endif
}

// ============================================================================
// Householder Reflections
// ============================================================================

MATDEF mat_elem_t mat_householder(Vec *v, mat_elem_t *tau, const Vec *x) {
  MAT_ASSERT_MAT(v);
  MAT_ASSERT_MAT(x);
  MAT_ASSERT(v->rows == x->rows);

  size_t n = x->rows;
  if (n == 0) {
    *tau = 0;
    return 0;
  }

  mat_elem_t *vd = v->data;
  const mat_elem_t *xd = x->data;

  // Compute ||x||
  mat_elem_t norm_sq = 0;
#ifdef MAT_HAS_ARM_NEON
  size_t i = 0;
  MAT_NEON_TYPE vsum = MAT_NEON_DUP(0);
  for (; i + MAT_NEON_WIDTH <= n; i += MAT_NEON_WIDTH) {
    MAT_NEON_TYPE xi = MAT_NEON_LOAD(&xd[i]);
    vsum = MAT_NEON_FMA(vsum, xi, xi);
  }
  norm_sq = MAT_NEON_ADDV(vsum);
  for (; i < n; i++) {
    norm_sq += xd[i] * xd[i];
  }
#else
  for (size_t i = 0; i < n; i++) {
    norm_sq += xd[i] * xd[i];
  }
#endif

  mat_elem_t norm_x = MAT_SQRT(norm_sq);

  if (norm_x < MAT_DEFAULT_EPSILON) {
    *tau = 0;
    for (size_t i = 0; i < n; i++)
      vd[i] = 0;
    return 0;
  }

  // Choose sign to avoid cancellation: beta = -sign(x[0]) * ||x||
  mat_elem_t x0 = xd[0];
  mat_elem_t beta = (x0 >= 0) ? -norm_x : norm_x;

  // v = x - beta*e1, so v[0] = x[0] - beta, v[i] = x[i] for i > 0
  mat_elem_t v0 = x0 - beta;

  // Normalize so v[0] = 1: v[i] = x[i] / v0 for i > 0
  mat_elem_t v0_inv = 1 / v0;
  vd[0] = 1;

#ifdef MAT_HAS_ARM_NEON
  MAT_NEON_TYPE vv0_inv = MAT_NEON_DUP(v0_inv);
  i = 1;
  for (; i + MAT_NEON_WIDTH <= n; i += MAT_NEON_WIDTH) {
    MAT_NEON_TYPE xi = MAT_NEON_LOAD(&xd[i]);
    MAT_NEON_STORE(&vd[i], MAT_NEON_MUL(xi, vv0_inv));
  }
  for (; i < n; i++) {
    vd[i] = xd[i] * v0_inv;
  }
#else
  for (size_t i = 1; i < n; i++) {
    vd[i] = xd[i] * v0_inv;
  }
#endif

  // tau = 2 / (v^T * v) = 2 / (1 + sum(v[i]^2 for i > 0))
  // = 2 / (1 + sum(x[i]^2 for i > 0) / v0^2)
  // = 2 * v0^2 / (v0^2 + sum(x[i]^2 for i > 0))
  // = 2 * v0^2 / (v0^2 + norm_sq - x0^2)
  // = 2 * v0^2 / ((x0 - beta)^2 + norm_sq - x0^2)
  // = 2 * v0^2 / (x0^2 - 2*x0*beta + beta^2 + norm_sq - x0^2)
  // = 2 * v0^2 / (beta^2 - 2*x0*beta + norm_sq)
  // Since beta^2 = norm_sq: = 2 * v0^2 / (2*norm_sq - 2*x0*beta)
  // = v0^2 / (norm_sq - x0*beta) = (x0 - beta)^2 / (norm_sq - x0*beta)
  // Simpler: tau = -v0 / beta (standard formula)
  *tau = -v0 / beta;

  return beta;
}

MATDEF void mat_householder_left(Mat *A, const Vec *v, mat_elem_t tau) {
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(v);
  MAT_ASSERT(A->rows == v->rows);

  if (tau == 0)
    return;

  // A = A - tau * v * (v^T * A)
  // w = v^T * A = (A^T * v), then A -= tau * v * w^T

  size_t n = A->cols;
  Vec *w = mat_vec(n);

  mat_gemv_t(w, 1, A, v, 0); // w = A^T * v
  mat_ger(A, -tau, v, w);    // A -= tau * v * w^T

  MAT_FREE_MAT(w);
}

MATDEF void mat_householder_right(Mat *A, const Vec *v, mat_elem_t tau) {
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(v);
  MAT_ASSERT(A->cols == v->rows);

  if (tau == 0)
    return;

  // A = A - tau * (A * v) * v^T

  size_t m = A->rows;

  Vec *w = mat_vec(m);

  mat_gemv(w, 1, A, v, 0); // w = A * v
  mat_ger(A, -tau, w, v);  // A -= tau * w * v^T

  MAT_FREE_MAT(w);
}

// QR block size for switching to blocked algorithm
#ifndef MAT_QR_BLOCK_SIZE
#define MAT_QR_BLOCK_SIZE 16
#endif

#ifndef MAT_QR_BLOCK_THRESHOLD
#define MAT_QR_BLOCK_THRESHOLD 64
#endif

// Column-major: Build T matrix for compact WY representation
// Y is stored column-major: Y[i,j] = Y[j * ldy + i]
// Optimized: precompute Y^T * Y to avoid redundant dot products
MAT_INTERNAL_STATIC void mat_qr_build_T_colmajor_(mat_elem_t *T, size_t ldt,
                                                   const mat_elem_t *Y, size_t ldy,
                                                   const mat_elem_t *tau, size_t m,
                                                   size_t k) {
  // T is stored column-major: T[i,j] = T[j * ldt + i]
  for (size_t i = 0; i < k * k; i++)
    T[i] = 0;

  // Precompute YtY[p,j] = Y[:,p]^T * Y[:,j] for p < j
  // Only upper triangle needed, stored column-major
  mat_elem_t *YtY = (mat_elem_t *)MAT_MALLOC(k * k * sizeof(mat_elem_t));
  for (size_t j = 1; j < k; j++) {
    const mat_elem_t *Yj = &Y[j * ldy];
    for (size_t p = 0; p < j; p++) {
      const mat_elem_t *Yp = &Y[p * ldy];
      mat_elem_t dot = 0;
#ifdef MAT_HAS_ARM_NEON
      size_t r = 0;
      MAT_NEON_TYPE acc = MAT_NEON_DUP(0);
      for (; r + MAT_NEON_WIDTH <= m; r += MAT_NEON_WIDTH) {
        MAT_NEON_TYPE yp = MAT_NEON_LOAD(&Yp[r]);
        MAT_NEON_TYPE yj = MAT_NEON_LOAD(&Yj[r]);
        acc = MAT_NEON_FMA(acc, yp, yj);
      }
      dot = MAT_NEON_ADDV(acc);
      for (; r < m; r++)
        dot += Yp[r] * Yj[r];
#else
      for (size_t r = 0; r < m; r++)
        dot += Yp[r] * Yj[r];
#endif
      YtY[j * k + p] = dot;  // YtY[p,j] stored at col j, row p
    }
  }

  // Build T using precomputed dot products
  for (size_t j = 0; j < k; j++) {
    T[j * ldt + j] = tau[j];  // T[j,j] = tau[j]

    for (size_t i = 0; i < j; i++) {
      mat_elem_t sum = 0;
      for (size_t p = i; p < j; p++) {
        sum += T[p * ldt + i] * YtY[j * k + p];  // T[i,p] * YtY[p,j]
      }
      T[j * ldt + i] = -tau[j] * sum;  // T[i,j]
    }
  }

  MAT_FREE(YtY);
}

// Fast column-major transpose: Bt = B^T
// B is rows x cols (col-major: B[r,c] = B[c*rows + r])
// Bt is cols x rows (col-major: Bt[r,c] = Bt[c*cols + r])
// We want: Bt[r,c] = B[c,r] => Bt[c*cols + r] = B[r*rows + c]
MAT_INTERNAL_STATIC void
mat_transpose_colmajor_(mat_elem_t *Bt, const mat_elem_t *B, size_t rows,
                         size_t cols) {
  for (size_t r = 0; r < cols; r++) {    // r = row in Bt, col in B
    for (size_t c = 0; c < rows; c++) {  // c = col in Bt, row in B
      Bt[c * cols + r] = B[r * rows + c];
    }
  }
}

// Workspace structure for blocked QR apply functions
typedef struct {
  mat_elem_t *W_mk;   // m × k workspace (for C, CT in apply_right)
  mat_elem_t *W_mk2;  // m × k workspace (second buffer)
  mat_elem_t *W_kn;   // k × n workspace (for C in apply_left)
  mat_elem_t *W_kn2;  // k × n workspace (for TC in apply_left)
} mat_qr_workspace_t;

// Column-major: Apply block Householder from left: A = (I - Y*T^T*Y^T) * A
// Uses pre-allocated workspace
MAT_INTERNAL_STATIC void
mat_qr_apply_block_left_colmajor_(mat_elem_t *A_data, size_t lda,
                                   const mat_elem_t *Y, size_t ldy,
                                   const mat_elem_t *T, size_t ldt,
                                   size_t m, size_t k, size_t n,
                                   mat_qr_workspace_t *ws) {
  (void)lda;
  (void)ldy;
  (void)ldt;

  // A = A - Y * T^T * (Y^T * A)
  mat_elem_t *C = ws->W_kn;    // k × n
  mat_elem_t *TC = ws->W_kn2;  // k × n

  // C = Y^T * A (k x n) - use transA flag instead of explicit transpose
  mat_gemm_strided_colmajor_(C, k, 1.0f, Y, ldy, MAT_TRANS, A_data, lda,
                             MAT_NO_TRANS, k, m, n, 0.0f);

  // TC = T^T * C (k x n) - use transA flag instead of explicit transpose
  mat_gemm_strided_colmajor_(TC, k, 1.0f, T, ldt, MAT_TRANS, C, k, MAT_NO_TRANS,
                             k, k, n, 0.0f);

  // A = A - Y * TC (m x n)
  mat_gemm_strided_colmajor_(A_data, lda, -1.0f, Y, ldy, MAT_NO_TRANS, TC, k,
                             MAT_NO_TRANS, m, k, n, 1.0f);
}

// Column-major: Apply block Householder from right: A = A * (I - Y*T*Y^T)
// Uses pre-allocated workspace
MAT_INTERNAL_STATIC void
mat_qr_apply_block_right_colmajor_(mat_elem_t *A_data, size_t lda,
                                    const mat_elem_t *Y, size_t ldy,
                                    const mat_elem_t *T, size_t ldt,
                                    size_t m, size_t n, size_t k,
                                    mat_qr_workspace_t *ws) {
  (void)lda;
  (void)ldy;
  (void)ldt;

  // A = A - (A * Y) * T * Y^T
  // A is m x n, Y is n x k, T is k x k
  mat_elem_t *C = ws->W_mk;    // m × k
  mat_elem_t *CT = ws->W_mk2;  // m × k

  // C = A * Y (m x k)
  mat_gemm_strided_colmajor_(C, m, 1.0f, A_data, lda, MAT_NO_TRANS, Y, ldy,
                             MAT_NO_TRANS, m, n, k, 0.0f);

  // CT = C * T (m x k)
  mat_gemm_strided_colmajor_(CT, m, 1.0f, C, m, MAT_NO_TRANS, T, ldt,
                             MAT_NO_TRANS, m, k, k, 0.0f);

  // A = A - CT * Y^T (m x n) - use transB flag instead of explicit transpose
  mat_gemm_strided_colmajor_(A_data, lda, -1.0f, CT, m, MAT_NO_TRANS, Y, ldy,
                             MAT_TRANS, m, k, n, 1.0f);
}

// Build T matrix for compact WY representation: H1*H2*...*Hk = I - Y*T*Y^T
// T is upper triangular with T[j,j] = tau[j]
// Row-major version: Y[i,j] = Y[i * ldy + j], T[i,j] = T[i * ldt + j]
MAT_INTERNAL_STATIC void mat_qr_build_T_(mat_elem_t *T, size_t ldt,
                                         const mat_elem_t *Y, size_t ldy,
                                         const mat_elem_t *tau, size_t m,
                                         size_t k) {
  for (size_t i = 0; i < k * k; i++)
    T[i] = 0;

  for (size_t j = 0; j < k; j++) {
    T[j * ldt + j] = tau[j];

    for (size_t i = 0; i < j; i++) {
      mat_elem_t sum = 0;
      for (size_t p = i; p < j; p++) {
        mat_elem_t dot = 0;
        for (size_t r = 0; r < m; r++) {
          dot += Y[r * ldy + p] * Y[r * ldy + j];
        }
        sum += T[i * ldt + p] * dot;
      }
      T[i * ldt + j] = -tau[j] * sum;
    }
  }
}

// Apply block Householder from left: A = (I - Y*T^T*Y^T) * A
// Uses T^T to get H_k * ... * H_1 ordering for QR
MAT_INTERNAL_STATIC void
mat_qr_apply_block_left_(mat_elem_t *A_data, size_t lda, const mat_elem_t *Y,
                         size_t ldy, const mat_elem_t *T, size_t ldt, size_t m,
                         size_t k, size_t n) {
  (void)lda; (void)ldy; (void)ldt;
  Mat Y_mat = {.rows = m, .cols = k, .data = (mat_elem_t *)Y};
  Mat A_mat = {.rows = m, .cols = n, .data = A_data};

  Mat *Yt = mat_rt(&Y_mat);
  Mat *C = mat_rmul(Yt, &A_mat);

  Mat T_mat = {.rows = k, .cols = k, .data = (mat_elem_t *)T};
  Mat *Tt = mat_rt(&T_mat);
  Mat *TC = mat_rmul(Tt, C);

  mat_gemm(&A_mat, -1, &Y_mat, TC, 1);

  mat_free_mat(Yt);
  mat_free_mat(C);
  mat_free_mat(Tt);
  mat_free_mat(TC);
}

// Apply block Householder from right: A = A * (I - Y*T*Y^T)
// Uses T to get H_1 * ... * H_k ordering for Q accumulation
MAT_INTERNAL_STATIC void
mat_qr_apply_block_right_(mat_elem_t *A_data, size_t lda, const mat_elem_t *Y,
                          size_t ldy, const mat_elem_t *T, size_t ldt, size_t m,
                          size_t n, size_t k) {
  (void)lda; (void)ldy; (void)ldt;
  Mat A_mat = {.rows = m, .cols = n, .data = A_data};
  Mat Y_mat = {.rows = n, .cols = k, .data = (mat_elem_t *)Y};
  Mat T_mat = {.rows = k, .cols = k, .data = (mat_elem_t *)T};

  Mat *C = mat_rmul(&A_mat, &Y_mat);
  Mat *CT = mat_rmul(C, &T_mat);
  Mat *Yt = mat_rt(&Y_mat);

  mat_gemm(&A_mat, -1, CT, Yt, 1);

  mat_free_mat(C);
  mat_free_mat(CT);
  mat_free_mat(Yt);
}

// QR Decomposition using Householder reflections
// A (m x n) = Q (m x m) * R (m x n), requires m >= n
// Uses blocked algorithm with BLAS-3 operations for large matrices
MATDEF void mat_qr(const Mat *A, Mat *Q, Mat *R) {
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(Q);
  MAT_ASSERT_MAT(R);

  size_t m = A->rows;
  size_t n = A->cols;
  MAT_ASSERT(m >= n && "mat_qr requires m >= n");
  MAT_ASSERT(Q->rows == m && Q->cols == m);
  MAT_ASSERT(R->rows == m && R->cols == n);

  mat_deep_copy(R, A);
  mat_eye(Q);

  // Column-major implementation: columns are contiguous
  // R[row, col] = R->data[col * m + row]
  // Q[row, col] = Q->data[col * m + row]

  // Use blocked algorithm for large matrices
  if (n >= MAT_QR_BLOCK_THRESHOLD) {
    size_t block_size = MAT_QR_BLOCK_SIZE;

    // Pre-allocate all workspace (using heap - scratch arena is reset by GEMM)
    mat_elem_t *tau_block =
        (mat_elem_t *)MAT_MALLOC(block_size * sizeof(mat_elem_t));
    mat_elem_t *Y =
        (mat_elem_t *)MAT_MALLOC(m * block_size * sizeof(mat_elem_t));
    mat_elem_t *T =
        (mat_elem_t *)MAT_MALLOC(block_size * block_size * sizeof(mat_elem_t));
    mat_elem_t *x_data = (mat_elem_t *)MAT_MALLOC(m * sizeof(mat_elem_t));

    // Pre-allocate workspace for apply functions (no transpose buffers needed)
    mat_qr_workspace_t ws;
    ws.W_mk = (mat_elem_t *)MAT_MALLOC(m * block_size * sizeof(mat_elem_t));
    ws.W_mk2 = (mat_elem_t *)MAT_MALLOC(m * block_size * sizeof(mat_elem_t));
    ws.W_kn = (mat_elem_t *)MAT_MALLOC(block_size * n * sizeof(mat_elem_t));
    ws.W_kn2 = (mat_elem_t *)MAT_MALLOC(block_size * n * sizeof(mat_elem_t));

    for (size_t jb = 0; jb < n; jb += block_size) {
      size_t kb = (jb + block_size <= n) ? block_size : (n - jb);
      size_t len = m - jb;

      // Panel factorization: compute kb Householder vectors
      for (size_t j = 0; j < kb; j++) {
        size_t col = jb + j;
        size_t vlen = m - col;

        // Extract column (x_data reused each iteration)
        for (size_t i = 0; i < vlen; i++) {
          x_data[i] = R->data[col * m + (col + i)];
        }

        Vec x_sub = {.rows = vlen, .cols = 1, .data = x_data};
        Vec v_sub = {.rows = vlen, .cols = 1, .data = x_data};
        (void)mat_householder(&v_sub, &tau_block[j], &x_sub);

        // Store v in Y (column-major): Y[row, j] = Y[j * len + row]
        size_t y_offset = col - jb;
        for (size_t i = 0; i < y_offset; i++) {
          Y[j * len + i] = 0;
        }
        for (size_t i = 0; i < vlen; i++) {
          Y[j * len + (y_offset + i)] = v_sub.data[i];
        }

        // Apply single Householder to remaining panel columns
        if (tau_block[j] != 0) {
          for (size_t c = col; c < jb + kb; c++) {
            Vec r_col = {.rows = vlen, .cols = 1, .data = &R->data[c * m + col]};
            mat_elem_t w = mat_dot(&v_sub, &r_col) * tau_block[j];
            mat_axpy(&r_col, -w, &v_sub);
          }
        }
      }

      // Build T matrix for WY representation (column-major)
      mat_qr_build_T_colmajor_(T, kb, Y, len, tau_block, len, kb);

      // Apply block reflector to trailing R (work directly with stride m)
      if (jb + kb < n) {
        size_t trail_cols = n - (jb + kb);
        mat_elem_t *R_src = &R->data[(jb + kb) * m + jb];

        mat_qr_apply_block_left_colmajor_(R_src, m, Y, len, T, kb, len, kb,
                                          trail_cols, &ws);
      }

      // Apply block reflector to Q (in-place - Q columns are always dense)
      // Q[:, jb:m] starts at Q->data[jb*m], all m rows accessed
      {
        mat_elem_t *Q_trail = &Q->data[jb * m];
        mat_qr_apply_block_right_colmajor_(Q_trail, m, Y, len, T, kb, m, len,
                                           kb, &ws);
      }
    }

    // Free workspace
    MAT_FREE(tau_block);
    MAT_FREE(Y);
    MAT_FREE(T);
    MAT_FREE(x_data);
    MAT_FREE(ws.W_mk);
    MAT_FREE(ws.W_mk2);
    MAT_FREE(ws.W_kn);
    MAT_FREE(ws.W_kn2);
    return;
  }

  // Unblocked algorithm for small matrices (column-major)
  mat_elem_t *x_data = (mat_elem_t *)MAT_MALLOC(m * sizeof(mat_elem_t));
  mat_elem_t *v_data = (mat_elem_t *)MAT_MALLOC(m * sizeof(mat_elem_t));
  mat_elem_t *u = (mat_elem_t *)MAT_MALLOC(m * sizeof(mat_elem_t));
  MAT_ASSERT(x_data && v_data && u);

  for (size_t j = 0; j < n; j++) {
    size_t len = m - j;

    // Extract column j, rows j to m-1 (column-major: contiguous!)
    // Use memcpy for contiguous data
    memcpy(x_data, &R->data[j * m + j], len * sizeof(mat_elem_t));

    Vec x_sub = {.rows = len, .cols = 1, .data = x_data};
    Vec v_sub = {.rows = len, .cols = 1, .data = v_data};

    mat_elem_t tau;
    (void)mat_householder(&v_sub, &tau, &x_sub);

    if (tau == 0)
      continue;

    // Apply H to R[j:m, j:n] from left (column-major)
    // For each column k, compute w = v^T * R[j:m, k], then R[:,k] -= w*v
    for (size_t k = j; k < n; k++) {
      mat_elem_t *Rk = &R->data[k * m + j];  // column k, starting at row j
      mat_elem_t w = 0;
#ifdef MAT_HAS_ARM_NEON
      size_t i = 0;
      MAT_NEON_TYPE wv = MAT_NEON_DUP(0);
      for (; i + MAT_NEON_WIDTH <= len; i += MAT_NEON_WIDTH) {
        MAT_NEON_TYPE vi = MAT_NEON_LOAD(&v_data[i]);
        MAT_NEON_TYPE ri = MAT_NEON_LOAD(&Rk[i]);
        wv = MAT_NEON_FMA(wv, vi, ri);
      }
      w = MAT_NEON_ADDV(wv);
      for (; i < len; i++)
        w += v_data[i] * Rk[i];
#else
      for (size_t i = 0; i < len; i++)
        w += v_data[i] * Rk[i];
#endif
      w *= tau;
#ifdef MAT_HAS_ARM_NEON
      MAT_NEON_TYPE wv2 = MAT_NEON_DUP(w);
      i = 0;
      for (; i + MAT_NEON_WIDTH <= len; i += MAT_NEON_WIDTH) {
        MAT_NEON_TYPE vi = MAT_NEON_LOAD(&v_data[i]);
        MAT_NEON_TYPE ri = MAT_NEON_LOAD(&Rk[i]);
        MAT_NEON_STORE(&Rk[i], MAT_NEON_FMS(ri, wv2, vi));
      }
      for (; i < len; i++)
        Rk[i] -= w * v_data[i];
#else
      for (size_t i = 0; i < len; i++)
        Rk[i] -= w * v_data[i];
#endif
    }

    // Apply H to Q[:, j:m] from right (column-major)
    // Q = Q * H = Q - tau * (Q * v) * v^T
    // Step 1: u = Q[:, j:m] * v (compute u column-by-column for contiguous access)
    memset(u, 0, m * sizeof(mat_elem_t));

    // u = Q[:, j:m] * v, computed column-by-column
    for (size_t jj = 0; jj < len; jj++) {
      mat_elem_t vjj = v_data[jj];
      mat_elem_t *Qcol = &Q->data[(j + jj) * m];  // column j+jj (contiguous)
#ifdef MAT_HAS_ARM_NEON
      MAT_NEON_TYPE vv = MAT_NEON_DUP(vjj);
      size_t i = 0;
      for (; i + MAT_NEON_WIDTH <= m; i += MAT_NEON_WIDTH) {
        MAT_NEON_TYPE ui = MAT_NEON_LOAD(&u[i]);
        MAT_NEON_TYPE qi = MAT_NEON_LOAD(&Qcol[i]);
        MAT_NEON_STORE(&u[i], MAT_NEON_FMA(ui, qi, vv));
      }
      for (; i < m; i++)
        u[i] += Qcol[i] * vjj;
#else
      for (size_t i = 0; i < m; i++)
        u[i] += Qcol[i] * vjj;
#endif
    }

    // Step 2: Q[:, j:m] -= tau * u * v^T, applied column by column
    for (size_t jj = 0; jj < len; jj++) {
      mat_elem_t scale = tau * v_data[jj];
      mat_elem_t *Qcol = &Q->data[(j + jj) * m];  // column j+jj (contiguous)
#ifdef MAT_HAS_ARM_NEON
      MAT_NEON_TYPE sv = MAT_NEON_DUP(scale);
      size_t i = 0;
      for (; i + MAT_NEON_WIDTH <= m; i += MAT_NEON_WIDTH) {
        MAT_NEON_TYPE qi = MAT_NEON_LOAD(&Qcol[i]);
        MAT_NEON_TYPE ui = MAT_NEON_LOAD(&u[i]);
        MAT_NEON_STORE(&Qcol[i], MAT_NEON_FMS(qi, sv, ui));
      }
      for (; i < m; i++)
        Qcol[i] -= scale * u[i];
#else
      for (size_t i = 0; i < m; i++)
        Qcol[i] -= scale * u[i];
#endif
    }
  }

  MAT_FREE(x_data);
  MAT_FREE(v_data);
  MAT_FREE(u);
}

// QR Decomposition - R factor only (skips Q computation)
// Much faster when only R is needed (e.g., least squares, rank determination)
MATDEF void mat_qr_r(const Mat *A, Mat *R) {
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(R);

  size_t m = A->rows;
  size_t n = A->cols;
  MAT_ASSERT(m >= n && "mat_qr_r requires m >= n");
  MAT_ASSERT(R->rows == m && R->cols == n);

  mat_deep_copy(R, A);

  // Column-major: use blocked algorithm for large matrices
  if (n >= MAT_QR_BLOCK_THRESHOLD) {
    size_t block_size = MAT_QR_BLOCK_SIZE;

    // Pre-allocate workspace (using heap - scratch arena is reset by GEMM)
    mat_elem_t *tau_block =
        (mat_elem_t *)MAT_MALLOC(block_size * sizeof(mat_elem_t));
    mat_elem_t *Y =
        (mat_elem_t *)MAT_MALLOC(m * block_size * sizeof(mat_elem_t));
    mat_elem_t *T =
        (mat_elem_t *)MAT_MALLOC(block_size * block_size * sizeof(mat_elem_t));
    mat_elem_t *x_data = (mat_elem_t *)MAT_MALLOC(m * sizeof(mat_elem_t));

    // Reduced workspace - no Q updates, no transpose buffers, no R_trail copy
    mat_elem_t *W_kn = (mat_elem_t *)MAT_MALLOC(block_size * n * sizeof(mat_elem_t));
    mat_elem_t *W_kn2 = (mat_elem_t *)MAT_MALLOC(block_size * n * sizeof(mat_elem_t));

    for (size_t jb = 0; jb < n; jb += block_size) {
      size_t kb = (jb + block_size <= n) ? block_size : (n - jb);
      size_t len = m - jb;

      // Panel factorization: compute kb Householder vectors
      for (size_t j = 0; j < kb; j++) {
        size_t col = jb + j;
        size_t vlen = m - col;

        for (size_t i = 0; i < vlen; i++) {
          x_data[i] = R->data[col * m + (col + i)];
        }

        Vec x_sub = {.rows = vlen, .cols = 1, .data = x_data};
        Vec v_sub = {.rows = vlen, .cols = 1, .data = x_data};
        (void)mat_householder(&v_sub, &tau_block[j], &x_sub);

        size_t y_offset = col - jb;
        for (size_t i = 0; i < y_offset; i++) {
          Y[j * len + i] = 0;
        }
        for (size_t i = 0; i < vlen; i++) {
          Y[j * len + (y_offset + i)] = v_sub.data[i];
        }

        if (tau_block[j] != 0) {
          for (size_t c = col; c < jb + kb; c++) {
            Vec r_col = {.rows = vlen, .cols = 1, .data = &R->data[c * m + col]};
            mat_elem_t w = mat_dot(&v_sub, &r_col) * tau_block[j];
            mat_axpy(&r_col, -w, &v_sub);
          }
        }
      }

      // Build T matrix for WY representation
      mat_qr_build_T_colmajor_(T, kb, Y, len, tau_block, len, kb);

      // Apply block reflector to trailing R only (skip Q!)
      // Work directly on R with stride m (no copy needed)
      if (jb + kb < n) {
        size_t trail_cols = n - (jb + kb);
        mat_elem_t *R_src = &R->data[(jb + kb) * m + jb];

        mat_elem_t *C = W_kn;
        mat_elem_t *TC = W_kn2;

        // C = Y^T * R_src (kb x trail_cols), R_src has stride m
        mat_gemm_strided_colmajor_(C, kb, 1.0f, Y, len, MAT_TRANS, R_src, m,
                                   MAT_NO_TRANS, kb, len, trail_cols, 0.0f);
        // TC = T^T * C (kb x trail_cols)
        mat_gemm_strided_colmajor_(TC, kb, 1.0f, T, kb, MAT_TRANS, C, kb,
                                   MAT_NO_TRANS, kb, kb, trail_cols, 0.0f);
        // R_src -= Y * TC (len x trail_cols), R_src has stride m
        mat_gemm_strided_colmajor_(R_src, m, -1.0f, Y, len, MAT_NO_TRANS, TC,
                                   kb, MAT_NO_TRANS, len, kb, trail_cols, 1.0f);
      }
      // Skip Q update entirely!
    }

    // Free workspace
    MAT_FREE(tau_block);
    MAT_FREE(Y);
    MAT_FREE(T);
    MAT_FREE(x_data);
    MAT_FREE(W_kn);
    MAT_FREE(W_kn2);
    return;
  }

  // Unblocked algorithm for small matrices (column-major)
  mat_elem_t *x_data = (mat_elem_t *)MAT_MALLOC(m * sizeof(mat_elem_t));
  mat_elem_t *v_data = (mat_elem_t *)MAT_MALLOC(m * sizeof(mat_elem_t));
  MAT_ASSERT(x_data && v_data);

  for (size_t j = 0; j < n; j++) {
    size_t len = m - j;
    memcpy(x_data, &R->data[j * m + j], len * sizeof(mat_elem_t));

    Vec x_sub = {.rows = len, .cols = 1, .data = x_data};
    Vec v_sub = {.rows = len, .cols = 1, .data = v_data};

    mat_elem_t tau;
    (void)mat_householder(&v_sub, &tau, &x_sub);

    if (tau == 0)
      continue;

    // Apply H to R[j:m, j:n] from left only
    for (size_t k = j; k < n; k++) {
      mat_elem_t *Rk = &R->data[k * m + j];
      mat_elem_t w = 0;
#ifdef MAT_HAS_ARM_NEON
      size_t i = 0;
      MAT_NEON_TYPE wv = MAT_NEON_DUP(0);
      for (; i + MAT_NEON_WIDTH <= len; i += MAT_NEON_WIDTH) {
        MAT_NEON_TYPE vi = MAT_NEON_LOAD(&v_data[i]);
        MAT_NEON_TYPE ri = MAT_NEON_LOAD(&Rk[i]);
        wv = MAT_NEON_FMA(wv, vi, ri);
      }
      w = MAT_NEON_ADDV(wv);
      for (; i < len; i++)
        w += v_data[i] * Rk[i];
#else
      for (size_t i = 0; i < len; i++)
        w += v_data[i] * Rk[i];
#endif
      w *= tau;
#ifdef MAT_HAS_ARM_NEON
      MAT_NEON_TYPE wv2 = MAT_NEON_DUP(w);
      i = 0;
      for (; i + MAT_NEON_WIDTH <= len; i += MAT_NEON_WIDTH) {
        MAT_NEON_TYPE vi = MAT_NEON_LOAD(&v_data[i]);
        MAT_NEON_TYPE ri = MAT_NEON_LOAD(&Rk[i]);
        MAT_NEON_STORE(&Rk[i], MAT_NEON_FMS(ri, wv2, vi));
      }
      for (; i < len; i++)
        Rk[i] -= w * v_data[i];
#else
      for (size_t i = 0; i < len; i++)
        Rk[i] -= w * v_data[i];
#endif
    }
    // Skip Q update!
  }

  MAT_FREE(x_data);
  MAT_FREE(v_data);
}

// Blocked partial pivoting LU (P * A = L * U)
// Uses cache-friendly blocking; compiler auto-vectorizes inner loops
#define MAT_PLU_BLOCK_SIZE 24

// Column-major blocked partial pivoting LU (P * A = L * U)
// Uses delayed pivoting: swap panel columns during factorization,
// then bulk-apply swaps to trailing columns before GEMM
MAT_INTERNAL_STATIC int mat_plu_blocked_colmajor_(Mat *M, Perm *p) {
  size_t n = M->rows;
  mat_elem_t *data = M->data;
  size_t *row_perm = p->data;
  int swap_count = 0;

  size_t pivot_rows[MAT_PLU_BLOCK_SIZE];

  for (size_t kb = 0; kb < n; kb += MAT_PLU_BLOCK_SIZE) {
    size_t k_end = (kb + MAT_PLU_BLOCK_SIZE < n) ? kb + MAT_PLU_BLOCK_SIZE : n;
    size_t block_k = k_end - kb;

    // Panel factorization with delayed pivoting
    for (size_t k = kb; k < k_end && k < n - 1; k++) {
      // Find pivot in column k (column is contiguous in col-major)
      mat_elem_t *col_k = &data[k * n];
      mat_elem_t max_val = MAT_FABS(col_k[k]);
      size_t pivot_row = k;
      for (size_t i = k + 1; i < n; i++) {
        mat_elem_t val = MAT_FABS(col_k[i]);
        if (val > max_val) {
          max_val = val;
          pivot_row = i;
        }
      }

      pivot_rows[k - kb] = pivot_row;

      // Swap rows only in columns 0:k_end (panel + already factored)
      if (pivot_row != k) {
        for (size_t j = 0; j < k_end; j++) {
          mat_elem_t *col_j = &data[j * n];
          mat_elem_t tmp = col_j[k];
          col_j[k] = col_j[pivot_row];
          col_j[pivot_row] = tmp;
        }
        size_t tmp = row_perm[k];
        row_perm[k] = row_perm[pivot_row];
        row_perm[pivot_row] = tmp;
        swap_count++;
      }

      if (MAT_FABS(col_k[k]) < MAT_DEFAULT_EPSILON)
        continue;

      mat_elem_t pivot_inv = 1.0f / col_k[k];

      // Scale L column (contiguous) - SIMD
      {
        size_t len = n - (k + 1);
        mat_elem_t *dst = &col_k[k + 1];
        MAT_NEON_TYPE vinv = MAT_NEON_DUP(pivot_inv);
        size_t i = 0;
        for (; i + MAT_NEON_WIDTH * 4 <= len; i += MAT_NEON_WIDTH * 4) {
          MAT_NEON_STORE(&dst[i], MAT_NEON_MUL(vinv, MAT_NEON_LOAD(&dst[i])));
          MAT_NEON_STORE(&dst[i + MAT_NEON_WIDTH], MAT_NEON_MUL(vinv, MAT_NEON_LOAD(&dst[i + MAT_NEON_WIDTH])));
          MAT_NEON_STORE(&dst[i + MAT_NEON_WIDTH * 2], MAT_NEON_MUL(vinv, MAT_NEON_LOAD(&dst[i + MAT_NEON_WIDTH * 2])));
          MAT_NEON_STORE(&dst[i + MAT_NEON_WIDTH * 3], MAT_NEON_MUL(vinv, MAT_NEON_LOAD(&dst[i + MAT_NEON_WIDTH * 3])));
        }
        for (; i < len; i++) dst[i] *= pivot_inv;
      }

      // Update panel columns k+1:k_end for all rows below k - SIMD
      for (size_t j = k + 1; j < k_end; j++) {
        mat_elem_t *col_j = &data[j * n];
        mat_elem_t u_kj = col_j[k];
        size_t len = n - (k + 1);
        mat_elem_t *dst = &col_j[k + 1];
        mat_elem_t *src = &col_k[k + 1];
        MAT_NEON_TYPE vu = MAT_NEON_DUP(u_kj);
        size_t i = 0;
        for (; i + MAT_NEON_WIDTH * 4 <= len; i += MAT_NEON_WIDTH * 4) {
          MAT_NEON_TYPE d0 = MAT_NEON_LOAD(&dst[i]);
          MAT_NEON_TYPE d1 = MAT_NEON_LOAD(&dst[i + MAT_NEON_WIDTH]);
          MAT_NEON_TYPE d2 = MAT_NEON_LOAD(&dst[i + MAT_NEON_WIDTH * 2]);
          MAT_NEON_TYPE d3 = MAT_NEON_LOAD(&dst[i + MAT_NEON_WIDTH * 3]);
          d0 = MAT_NEON_SUB(d0, MAT_NEON_MUL(vu, MAT_NEON_LOAD(&src[i])));
          d1 = MAT_NEON_SUB(d1, MAT_NEON_MUL(vu, MAT_NEON_LOAD(&src[i + MAT_NEON_WIDTH])));
          d2 = MAT_NEON_SUB(d2, MAT_NEON_MUL(vu, MAT_NEON_LOAD(&src[i + MAT_NEON_WIDTH * 2])));
          d3 = MAT_NEON_SUB(d3, MAT_NEON_MUL(vu, MAT_NEON_LOAD(&src[i + MAT_NEON_WIDTH * 3])));
          MAT_NEON_STORE(&dst[i], d0);
          MAT_NEON_STORE(&dst[i + MAT_NEON_WIDTH], d1);
          MAT_NEON_STORE(&dst[i + MAT_NEON_WIDTH * 2], d2);
          MAT_NEON_STORE(&dst[i + MAT_NEON_WIDTH * 3], d3);
        }
        for (; i < len; i++) dst[i] -= u_kj * src[i];
      }
    }

    // Apply delayed pivots to trailing columns and compute U12
    if (k_end < n) {
      // 1. Apply panel pivots to trailing columns
      for (size_t k = kb; k < k_end && k < n - 1; k++) {
        size_t pivot_row = pivot_rows[k - kb];
        if (pivot_row != k) {
          for (size_t j = k_end; j < n; j++) {
            mat_elem_t *col_j = &data[j * n];
            mat_elem_t tmp = col_j[k];
            col_j[k] = col_j[pivot_row];
            col_j[pivot_row] = tmp;
          }
        }
      }

      // 2. TRSM: Solve L11 * U12 = A12 - SIMD
      for (size_t k = kb; k < k_end; k++) {
        mat_elem_t *col_k = &data[k * n];
        for (size_t j = k_end; j < n; j++) {
          mat_elem_t *col_j = &data[j * n];
          mat_elem_t u_kj = col_j[k];
          size_t len = k_end - (k + 1);
          mat_elem_t *dst = &col_j[k + 1];
          mat_elem_t *src = &col_k[k + 1];
          MAT_NEON_TYPE vu = MAT_NEON_DUP(u_kj);
          size_t i = 0;
          for (; i + MAT_NEON_WIDTH * 4 <= len; i += MAT_NEON_WIDTH * 4) {
            MAT_NEON_TYPE d0 = MAT_NEON_LOAD(&dst[i]);
            MAT_NEON_TYPE d1 = MAT_NEON_LOAD(&dst[i + MAT_NEON_WIDTH]);
            MAT_NEON_TYPE d2 = MAT_NEON_LOAD(&dst[i + MAT_NEON_WIDTH * 2]);
            MAT_NEON_TYPE d3 = MAT_NEON_LOAD(&dst[i + MAT_NEON_WIDTH * 3]);
            d0 = MAT_NEON_SUB(d0, MAT_NEON_MUL(vu, MAT_NEON_LOAD(&src[i])));
            d1 = MAT_NEON_SUB(d1, MAT_NEON_MUL(vu, MAT_NEON_LOAD(&src[i + MAT_NEON_WIDTH])));
            d2 = MAT_NEON_SUB(d2, MAT_NEON_MUL(vu, MAT_NEON_LOAD(&src[i + MAT_NEON_WIDTH * 2])));
            d3 = MAT_NEON_SUB(d3, MAT_NEON_MUL(vu, MAT_NEON_LOAD(&src[i + MAT_NEON_WIDTH * 3])));
            MAT_NEON_STORE(&dst[i], d0);
            MAT_NEON_STORE(&dst[i + MAT_NEON_WIDTH], d1);
            MAT_NEON_STORE(&dst[i + MAT_NEON_WIDTH * 2], d2);
            MAT_NEON_STORE(&dst[i + MAT_NEON_WIDTH * 3], d3);
          }
          for (; i < len; i++) dst[i] -= u_kj * src[i];
        }
      }

      // 3. GEMM: A22 -= L21 * U12
      size_t trail_m = n - k_end;
      size_t trail_n = n - k_end;

      mat_gemm_strided_colmajor_(
          &data[k_end * n + k_end], n,
          -1.0f,
          &data[kb * n + k_end], n, MAT_NO_TRANS,
          &data[k_end * n + kb], n, MAT_NO_TRANS,
          trail_m, block_k, trail_n,
          1.0f);
    }
  }

  return swap_count;
}

// Scalar implementation of full pivoting LU (P * A * Q = L * U)
MAT_INTERNAL_STATIC int mat_lu_scalar_(Mat *M, Perm *p, Perm *q) {
  size_t n = M->rows;
  mat_elem_t *data = M->data;
  size_t *row_perm = p->data;
  size_t *col_perm = q->data;
  int swap_count = 0;

  for (size_t k = 0; k < n; k++) {
    // Find largest element in submatrix data[k:n, k:n]
    mat_elem_t max_val = 0;
    size_t pivot_row = k, pivot_col = k;

    for (size_t i = k; i < n; i++) {
      for (size_t j = k; j < n; j++) {
        mat_elem_t val = MAT_FABS(data[i * n + j]);
        if (val > max_val) {
          max_val = val;
          pivot_row = i;
          pivot_col = j;
        }
      }
    }

    // Swap rows k and pivot_row
    if (pivot_row != k) {
      for (size_t j = 0; j < n; j++) {
        mat_elem_t tmp = data[k * n + j];
        data[k * n + j] = data[pivot_row * n + j];
        data[pivot_row * n + j] = tmp;
      }
      size_t tmp = row_perm[k];
      row_perm[k] = row_perm[pivot_row];
      row_perm[pivot_row] = tmp;
      swap_count++;
    }

    // Swap columns k and pivot_col
    if (pivot_col != k) {
      for (size_t i = 0; i < n; i++) {
        mat_elem_t tmp = data[i * n + k];
        data[i * n + k] = data[i * n + pivot_col];
        data[i * n + pivot_col] = tmp;
      }
      size_t tmp = col_perm[k];
      col_perm[k] = col_perm[pivot_col];
      col_perm[pivot_col] = tmp;
      swap_count++;
    }

    // Skip if pivot is zero (matrix is singular)
    if (MAT_FABS(data[k * n + k]) < MAT_DEFAULT_EPSILON) {
      continue;
    }

    // Elimination: rank-1 update of trailing submatrix
    mat_elem_t pivot_inv = 1.0f / data[k * n + k];
    mat_elem_t *row_k = &data[k * n];

    for (size_t i = k + 1; i < n; i++) {
      mat_elem_t *row_i = &data[i * n];
      mat_elem_t l_ik = row_i[k] * pivot_inv;
      row_i[k] = l_ik;

      for (size_t j = k + 1; j < n; j++) {
        row_i[j] -= l_ik * row_k[j];
      }
    }
  }

  return swap_count;
}

#ifdef MAT_HAS_ARM_NEON
// NEON-optimized implementation of full pivoting LU
MAT_INTERNAL_STATIC int mat_lu_neon_(Mat *M, Perm *p, Perm *q) {
  size_t n = M->rows;
  mat_elem_t *data = M->data;
  size_t *row_perm = p->data;
  size_t *col_perm = q->data;
  int swap_count = 0;

  for (size_t k = 0; k < n; k++) {
    // Find largest element in submatrix data[k:n, k:n] using NEON
    mat_elem_t max_val = 0;
    size_t pivot_row = k, pivot_col = k;

    for (size_t i = k; i < n; i++) {
      mat_elem_t *row = &data[i * n];
      size_t j = k;

      // NEON vectorized max search within row
      MAT_NEON_TYPE vmax = MAT_NEON_DUP(0);
      for (; j + MAT_NEON_WIDTH <= n; j += MAT_NEON_WIDTH) {
        MAT_NEON_TYPE v = MAT_NEON_LOAD(&row[j]);
        MAT_NEON_TYPE vabs = MAT_NEON_ABS(v);
        vmax = MAT_NEON_MAX(vmax, vabs);
      }
      mat_elem_t row_max = MAT_NEON_MAXV(vmax);

      // Scalar remainder
      for (; j < n; j++) {
        mat_elem_t val = MAT_FABS(row[j]);
        if (val > row_max)
          row_max = val;
      }

      // If this row has a larger max, find the exact column
      if (row_max > max_val) {
        max_val = row_max;
        pivot_row = i;
        // Find the column with max value in this row
        for (size_t jj = k; jj < n; jj++) {
          if (MAT_FABS(row[jj]) == row_max) {
            pivot_col = jj;
            break;
          }
        }
      }
    }

    // Swap rows k and pivot_row using NEON
    if (pivot_row != k) {
      mat_elem_t *row_k_ptr = &data[k * n];
      mat_elem_t *row_p_ptr = &data[pivot_row * n];
      size_t j = 0;

      for (; j + MAT_NEON_WIDTH <= n; j += MAT_NEON_WIDTH) {
        MAT_NEON_TYPE vk = MAT_NEON_LOAD(&row_k_ptr[j]);
        MAT_NEON_TYPE vp = MAT_NEON_LOAD(&row_p_ptr[j]);
        MAT_NEON_STORE(&row_k_ptr[j], vp);
        MAT_NEON_STORE(&row_p_ptr[j], vk);
      }
      for (; j < n; j++) {
        mat_elem_t tmp = row_k_ptr[j];
        row_k_ptr[j] = row_p_ptr[j];
        row_p_ptr[j] = tmp;
      }

      size_t tmp = row_perm[k];
      row_perm[k] = row_perm[pivot_row];
      row_perm[pivot_row] = tmp;
      swap_count++;
    }

    // Swap columns k and pivot_col (not vectorizable - strided access)
    if (pivot_col != k) {
      for (size_t i = 0; i < n; i++) {
        mat_elem_t tmp = data[i * n + k];
        data[i * n + k] = data[i * n + pivot_col];
        data[i * n + pivot_col] = tmp;
      }
      size_t tmp = col_perm[k];
      col_perm[k] = col_perm[pivot_col];
      col_perm[pivot_col] = tmp;
      swap_count++;
    }

    // Skip if pivot is zero
    if (MAT_FABS(data[k * n + k]) < MAT_DEFAULT_EPSILON) {
      continue;
    }

    // Elimination: rank-1 update using NEON
    mat_elem_t pivot_inv = 1.0f / data[k * n + k];
    mat_elem_t *row_k = &data[k * n];

    for (size_t i = k + 1; i < n; i++) {
      mat_elem_t *row_i = &data[i * n];
      mat_elem_t l_ik = row_i[k] * pivot_inv;
      row_i[k] = l_ik;

      MAT_NEON_TYPE vl = MAT_NEON_DUP(l_ik);
      size_t j = k + 1;

      // NEON vectorized update: row_i[j] -= l_ik * row_k[j]
      for (; j + MAT_NEON_WIDTH * 4 <= n; j += MAT_NEON_WIDTH * 4) {
        MAT_NEON_TYPE vk0 = MAT_NEON_LOAD(&row_k[j]);
        MAT_NEON_TYPE vk1 = MAT_NEON_LOAD(&row_k[j + MAT_NEON_WIDTH]);
        MAT_NEON_TYPE vk2 = MAT_NEON_LOAD(&row_k[j + MAT_NEON_WIDTH * 2]);
        MAT_NEON_TYPE vk3 = MAT_NEON_LOAD(&row_k[j + MAT_NEON_WIDTH * 3]);

        MAT_NEON_TYPE vi0 = MAT_NEON_LOAD(&row_i[j]);
        MAT_NEON_TYPE vi1 = MAT_NEON_LOAD(&row_i[j + MAT_NEON_WIDTH]);
        MAT_NEON_TYPE vi2 = MAT_NEON_LOAD(&row_i[j + MAT_NEON_WIDTH * 2]);
        MAT_NEON_TYPE vi3 = MAT_NEON_LOAD(&row_i[j + MAT_NEON_WIDTH * 3]);

        vi0 = MAT_NEON_FMS(vi0, vl, vk0);
        vi1 = MAT_NEON_FMS(vi1, vl, vk1);
        vi2 = MAT_NEON_FMS(vi2, vl, vk2);
        vi3 = MAT_NEON_FMS(vi3, vl, vk3);

        MAT_NEON_STORE(&row_i[j], vi0);
        MAT_NEON_STORE(&row_i[j + MAT_NEON_WIDTH], vi1);
        MAT_NEON_STORE(&row_i[j + MAT_NEON_WIDTH * 2], vi2);
        MAT_NEON_STORE(&row_i[j + MAT_NEON_WIDTH * 3], vi3);
      }

      for (; j + MAT_NEON_WIDTH <= n; j += MAT_NEON_WIDTH) {
        MAT_NEON_TYPE vk = MAT_NEON_LOAD(&row_k[j]);
        MAT_NEON_TYPE vi = MAT_NEON_LOAD(&row_i[j]);
        vi = MAT_NEON_FMS(vi, vl, vk);
        MAT_NEON_STORE(&row_i[j], vi);
      }

      // Scalar remainder
      for (; j < n; j++) {
        row_i[j] -= l_ik * row_k[j];
      }
    }
  }

  return swap_count;
}
#endif

#ifdef MAT_HAS_ARM_NEON
// Column-major NEON-optimized full pivoting LU decomposition
// In column-major: column j is contiguous at &data[j * n]
MAT_INTERNAL_STATIC int mat_lu_colmajor_neon_(Mat *M, Perm *p, Perm *q) {
  size_t n = M->rows;
  mat_elem_t *data = M->data;
  size_t *row_perm = p->data;
  size_t *col_perm = q->data;
  int swap_count = 0;

  for (size_t k = 0; k < n; k++) {
    // Find largest element in submatrix data[k:n, k:n]
    // Column-major: iterate by columns (contiguous), then rows
    mat_elem_t max_val = 0;
    size_t pivot_row = k, pivot_col = k;

    for (size_t j = k; j < n; j++) {
      mat_elem_t *col = &data[j * n];
      size_t i = k;

      // NEON vectorized max search within column (contiguous!)
      MAT_NEON_TYPE vmax = MAT_NEON_DUP(0);
      for (; i + MAT_NEON_WIDTH <= n; i += MAT_NEON_WIDTH) {
        MAT_NEON_TYPE v = MAT_NEON_LOAD(&col[i]);
        MAT_NEON_TYPE vabs = MAT_NEON_ABS(v);
        vmax = MAT_NEON_MAX(vmax, vabs);
      }
      mat_elem_t col_max = MAT_NEON_MAXV(vmax);

      // Scalar remainder
      for (; i < n; i++) {
        mat_elem_t val = MAT_FABS(col[i]);
        if (val > col_max)
          col_max = val;
      }

      // If this column has a larger max, find the exact row
      if (col_max > max_val) {
        max_val = col_max;
        pivot_col = j;
        // Find the row with max value in this column
        for (size_t ii = k; ii < n; ii++) {
          if (MAT_FABS(col[ii]) == col_max) {
            pivot_row = ii;
            break;
          }
        }
      }
    }

    // Swap columns k and pivot_col using NEON (columns are contiguous!)
    if (pivot_col != k) {
      mat_elem_t *col_k = &data[k * n];
      mat_elem_t *col_p = &data[pivot_col * n];
      size_t i = 0;

      for (; i + MAT_NEON_WIDTH * 4 <= n; i += MAT_NEON_WIDTH * 4) {
        MAT_NEON_TYPE vk0 = MAT_NEON_LOAD(&col_k[i]);
        MAT_NEON_TYPE vk1 = MAT_NEON_LOAD(&col_k[i + MAT_NEON_WIDTH]);
        MAT_NEON_TYPE vk2 = MAT_NEON_LOAD(&col_k[i + MAT_NEON_WIDTH * 2]);
        MAT_NEON_TYPE vk3 = MAT_NEON_LOAD(&col_k[i + MAT_NEON_WIDTH * 3]);
        MAT_NEON_TYPE vp0 = MAT_NEON_LOAD(&col_p[i]);
        MAT_NEON_TYPE vp1 = MAT_NEON_LOAD(&col_p[i + MAT_NEON_WIDTH]);
        MAT_NEON_TYPE vp2 = MAT_NEON_LOAD(&col_p[i + MAT_NEON_WIDTH * 2]);
        MAT_NEON_TYPE vp3 = MAT_NEON_LOAD(&col_p[i + MAT_NEON_WIDTH * 3]);
        MAT_NEON_STORE(&col_k[i], vp0);
        MAT_NEON_STORE(&col_k[i + MAT_NEON_WIDTH], vp1);
        MAT_NEON_STORE(&col_k[i + MAT_NEON_WIDTH * 2], vp2);
        MAT_NEON_STORE(&col_k[i + MAT_NEON_WIDTH * 3], vp3);
        MAT_NEON_STORE(&col_p[i], vk0);
        MAT_NEON_STORE(&col_p[i + MAT_NEON_WIDTH], vk1);
        MAT_NEON_STORE(&col_p[i + MAT_NEON_WIDTH * 2], vk2);
        MAT_NEON_STORE(&col_p[i + MAT_NEON_WIDTH * 3], vk3);
      }
      for (; i + MAT_NEON_WIDTH <= n; i += MAT_NEON_WIDTH) {
        MAT_NEON_TYPE vk = MAT_NEON_LOAD(&col_k[i]);
        MAT_NEON_TYPE vp = MAT_NEON_LOAD(&col_p[i]);
        MAT_NEON_STORE(&col_k[i], vp);
        MAT_NEON_STORE(&col_p[i], vk);
      }
      for (; i < n; i++) {
        mat_elem_t tmp = col_k[i];
        col_k[i] = col_p[i];
        col_p[i] = tmp;
      }

      size_t tmp = col_perm[k];
      col_perm[k] = col_perm[pivot_col];
      col_perm[pivot_col] = tmp;
      swap_count++;
    }

    // Swap rows k and pivot_row (strided access in column-major)
    if (pivot_row != k) {
      for (size_t j = 0; j < n; j++) {
        mat_elem_t tmp = data[j * n + k];
        data[j * n + k] = data[j * n + pivot_row];
        data[j * n + pivot_row] = tmp;
      }
      size_t tmp = row_perm[k];
      row_perm[k] = row_perm[pivot_row];
      row_perm[pivot_row] = tmp;
      swap_count++;
    }

    // Skip if pivot is zero
    mat_elem_t *col_k = &data[k * n];
    if (MAT_FABS(col_k[k]) < MAT_DEFAULT_EPSILON) {
      continue;
    }

    // Compute multipliers: L[i,k] = M[i,k] / M[k,k] for i > k
    mat_elem_t pivot_inv = 1.0f / col_k[k];
    for (size_t i = k + 1; i < n; i++) {
      col_k[i] *= pivot_inv;
    }

    // Elimination: column-oriented rank-1 update
    // For each column j > k: col_j[k+1:n] -= col_j[k] * col_k[k+1:n]
    for (size_t j = k + 1; j < n; j++) {
      mat_elem_t *col_j = &data[j * n];
      mat_elem_t scale = col_j[k];
      MAT_NEON_TYPE vscale = MAT_NEON_DUP(scale);

      size_t i = k + 1;
      // NEON vectorized update (both col_j and col_k are contiguous!)
      for (; i + MAT_NEON_WIDTH * 4 <= n; i += MAT_NEON_WIDTH * 4) {
        MAT_NEON_TYPE vk0 = MAT_NEON_LOAD(&col_k[i]);
        MAT_NEON_TYPE vk1 = MAT_NEON_LOAD(&col_k[i + MAT_NEON_WIDTH]);
        MAT_NEON_TYPE vk2 = MAT_NEON_LOAD(&col_k[i + MAT_NEON_WIDTH * 2]);
        MAT_NEON_TYPE vk3 = MAT_NEON_LOAD(&col_k[i + MAT_NEON_WIDTH * 3]);

        MAT_NEON_TYPE vj0 = MAT_NEON_LOAD(&col_j[i]);
        MAT_NEON_TYPE vj1 = MAT_NEON_LOAD(&col_j[i + MAT_NEON_WIDTH]);
        MAT_NEON_TYPE vj2 = MAT_NEON_LOAD(&col_j[i + MAT_NEON_WIDTH * 2]);
        MAT_NEON_TYPE vj3 = MAT_NEON_LOAD(&col_j[i + MAT_NEON_WIDTH * 3]);

        vj0 = MAT_NEON_FMS(vj0, vscale, vk0);
        vj1 = MAT_NEON_FMS(vj1, vscale, vk1);
        vj2 = MAT_NEON_FMS(vj2, vscale, vk2);
        vj3 = MAT_NEON_FMS(vj3, vscale, vk3);

        MAT_NEON_STORE(&col_j[i], vj0);
        MAT_NEON_STORE(&col_j[i + MAT_NEON_WIDTH], vj1);
        MAT_NEON_STORE(&col_j[i + MAT_NEON_WIDTH * 2], vj2);
        MAT_NEON_STORE(&col_j[i + MAT_NEON_WIDTH * 3], vj3);
      }

      for (; i + MAT_NEON_WIDTH <= n; i += MAT_NEON_WIDTH) {
        MAT_NEON_TYPE vk = MAT_NEON_LOAD(&col_k[i]);
        MAT_NEON_TYPE vj = MAT_NEON_LOAD(&col_j[i]);
        vj = MAT_NEON_FMS(vj, vscale, vk);
        MAT_NEON_STORE(&col_j[i], vj);
      }

      // Scalar remainder
      for (; i < n; i++) {
        col_j[i] -= scale * col_k[i];
      }
    }
  }

  return swap_count;
}
#endif

MATDEF int mat_plu(const Mat *A, Mat *L, Mat *U, Perm *p) {
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(L);
  MAT_ASSERT_MAT(U);
  MAT_ASSERT(p != NULL);
  MAT_ASSERT_SQUARE(A);

  size_t n = A->rows;
  MAT_ASSERT(L->rows == n && L->cols == n);
  MAT_ASSERT(U->rows == n && U->cols == n);
  MAT_ASSERT(p->size == n);

  Mat *M = mat_rdeep_copy(A);
  mat_perm_identity(p);

  // Column-major blocked PLU
  int swap_count = mat_plu_blocked_colmajor_(M, p);

  // Extract L and U from M (column-major result)
  // M contains L (below diagonal) and U (on and above diagonal)
  mat_eye(L);
  memset(U->data, 0, n * n * sizeof(mat_elem_t));

  mat_elem_t *data = M->data;
  for (size_t j = 0; j < n; j++) {
    mat_elem_t *Mj = &data[j * n];
    mat_elem_t *Lj = &L->data[j * n];
    mat_elem_t *Uj = &U->data[j * n];
    // L[i,j] for i > j (below diagonal)
    for (size_t i = j + 1; i < n; i++)
      Lj[i] = Mj[i];
    // U[i,j] for i <= j (on and above diagonal)
    for (size_t i = 0; i <= j; i++)
      Uj[i] = Mj[i];
  }

  MAT_FREE_MAT(M);
  return swap_count;
}

MATDEF int mat_lu(const Mat *A, Mat *L, Mat *U, Perm *p, Perm *q) {
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(L);
  MAT_ASSERT_MAT(U);
  MAT_ASSERT(p != NULL && q != NULL);

  size_t n = A->rows;
  MAT_ASSERT(A->cols == n && "mat_lu requires square matrix");
  MAT_ASSERT(L->rows == n && L->cols == n);
  MAT_ASSERT(U->rows == n && U->cols == n);
  MAT_ASSERT(p->size == n && q->size == n);

  Mat *M = mat_rdeep_copy(A);

  // Initialize permutations to identity
  mat_perm_identity(p);
  mat_perm_identity(q);

#ifdef MAT_HAS_ARM_NEON
  // Column-major LU decomposition
  int swap_count = mat_lu_colmajor_neon_(M, p, q);
#else
  // Scalar fallback - use simpler algorithm
  int swap_count = 0;
  mat_elem_t *data = M->data;
  for (size_t k = 0; k < n - 1; k++) {
    // Find pivot
    size_t pivot_row = k, pivot_col = k;
    mat_elem_t max_val = MAT_FABS(data[k * n + k]);
    for (size_t i = k; i < n; i++) {
      for (size_t j = k; j < n; j++) {
        mat_elem_t val = MAT_FABS(data[j * n + i]);
        if (val > max_val) {
          max_val = val;
          pivot_row = i;
          pivot_col = j;
        }
      }
    }

    if (max_val < MAT_DEFAULT_EPSILON) continue;

    // Swap rows in M and p
    if (pivot_row != k) {
      for (size_t j = 0; j < n; j++) {
        mat_elem_t tmp = data[j * n + k];
        data[j * n + k] = data[j * n + pivot_row];
        data[j * n + pivot_row] = tmp;
      }
      size_t tmp = p->data[k];
      p->data[k] = p->data[pivot_row];
      p->data[pivot_row] = tmp;
      swap_count++;
    }

    // Swap columns in M and q
    if (pivot_col != k) {
      for (size_t i = 0; i < n; i++) {
        mat_elem_t tmp = data[k * n + i];
        data[k * n + i] = data[pivot_col * n + i];
        data[pivot_col * n + i] = tmp;
      }
      size_t tmp = q->data[k];
      q->data[k] = q->data[pivot_col];
      q->data[pivot_col] = tmp;
      swap_count++;
    }

    // Elimination
    mat_elem_t pivot_inv = 1.0f / data[k * n + k];
    for (size_t i = k + 1; i < n; i++) {
      mat_elem_t l_ik = data[k * n + i] * pivot_inv;
      data[k * n + i] = l_ik;
      for (size_t j = k + 1; j < n; j++) {
        data[j * n + i] -= l_ik * data[j * n + k];
      }
    }
  }
#endif

  // Extract L and U from M (column-major result)
  mat_eye(L);
  memset(U->data, 0, n * n * sizeof(mat_elem_t));

  mat_elem_t *data = M->data;
  for (size_t j = 0; j < n; j++) {
    mat_elem_t *Mcol = &data[j * n];
    mat_elem_t *Lcol = &L->data[j * n];
    mat_elem_t *Ucol = &U->data[j * n];
    // L: copy below diagonal
    for (size_t i = j + 1; i < n; i++)
      Lcol[i] = Mcol[i];
    // U: copy diagonal and above
    for (size_t i = 0; i <= j; i++)
      Ucol[i] = Mcol[i];
  }

  MAT_FREE_MAT(M);
  return swap_count;
}

// ============================================================================
// Raw dot product (used by triangular solvers and Cholesky)
// ============================================================================

// Dot product of two raw arrays (NEON-optimized when available)
MAT_INTERNAL_STATIC mat_elem_t mat_dot_raw_(const mat_elem_t *a,
                                            const mat_elem_t *b, size_t n) {
#ifdef MAT_HAS_ARM_NEON
  MAT_NEON_TYPE vsum0 = MAT_NEON_DUP(0);
  MAT_NEON_TYPE vsum1 = MAT_NEON_DUP(0);
  MAT_NEON_TYPE vsum2 = MAT_NEON_DUP(0);
  MAT_NEON_TYPE vsum3 = MAT_NEON_DUP(0);

  size_t i = 0;
  for (; i + MAT_NEON_WIDTH * 4 <= n; i += MAT_NEON_WIDTH * 4) {
    MAT_NEON_TYPE va0 = MAT_NEON_LOAD(&a[i]);
    MAT_NEON_TYPE va1 = MAT_NEON_LOAD(&a[i + MAT_NEON_WIDTH]);
    MAT_NEON_TYPE va2 = MAT_NEON_LOAD(&a[i + MAT_NEON_WIDTH * 2]);
    MAT_NEON_TYPE va3 = MAT_NEON_LOAD(&a[i + MAT_NEON_WIDTH * 3]);

    MAT_NEON_TYPE vb0 = MAT_NEON_LOAD(&b[i]);
    MAT_NEON_TYPE vb1 = MAT_NEON_LOAD(&b[i + MAT_NEON_WIDTH]);
    MAT_NEON_TYPE vb2 = MAT_NEON_LOAD(&b[i + MAT_NEON_WIDTH * 2]);
    MAT_NEON_TYPE vb3 = MAT_NEON_LOAD(&b[i + MAT_NEON_WIDTH * 3]);

    vsum0 = MAT_NEON_FMA(vsum0, va0, vb0);
    vsum1 = MAT_NEON_FMA(vsum1, va1, vb1);
    vsum2 = MAT_NEON_FMA(vsum2, va2, vb2);
    vsum3 = MAT_NEON_FMA(vsum3, va3, vb3);
  }

  for (; i + MAT_NEON_WIDTH <= n; i += MAT_NEON_WIDTH) {
    MAT_NEON_TYPE va = MAT_NEON_LOAD(&a[i]);
    MAT_NEON_TYPE vb = MAT_NEON_LOAD(&b[i]);
    vsum0 = MAT_NEON_FMA(vsum0, va, vb);
  }

  vsum0 = MAT_NEON_ADD(vsum0, vsum1);
  vsum2 = MAT_NEON_ADD(vsum2, vsum3);
  vsum0 = MAT_NEON_ADD(vsum0, vsum2);
  mat_elem_t result = MAT_NEON_ADDV(vsum0);

  for (; i < n; i++) {
    result += a[i] * b[i];
  }
  return result;
#else
  mat_elem_t result = 0;
  for (size_t i = 0; i < n; i++) {
    result += a[i] * b[i];
  }
  return result;
#endif
}

// ============================================================================
// Triangular system solvers (TRSV)
// ============================================================================

// --- mat_solve_tril: Solve Lx = b, L lower triangular (non-unit diagonal) ---

MAT_INTERNAL_STATIC void mat_solve_tril_scalar_(Vec *x, const Mat *L,
                                                 const Vec *b) {
  size_t n = L->rows;
  mat_elem_t *x_data = x->data;
  const mat_elem_t *b_data = b->data;

  // Forward substitution: x[i] = (b[i] - sum(L[i,j]*x[j], j<i)) / L[i,i]
  for (size_t i = 0; i < n; i++) {
    mat_elem_t dot = 0;
    for (size_t j = 0; j < i; j++) {
      dot += MAT_AT(L, i, j) * x_data[j];
    }
    x_data[i] = (b_data[i] - dot) / MAT_AT(L, i, i);
  }
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void mat_solve_tril_neon_(Vec *x, const Mat *L,
                                               const Vec *b) {
  // Column-major: columns are contiguous, use column-oriented forward substitution
  // x[j] = x[j] / L[j,j], then update x[j+1:n] -= x[j] * L[j+1:n, j]
  size_t n = L->rows;
  const mat_elem_t *L_data = L->data;

  // Copy b to x (we'll modify x in place)
  memcpy(x->data, b->data, n * sizeof(mat_elem_t));

  for (size_t j = 0; j < n; j++) {
    const mat_elem_t *Lj = &L_data[j * n];  // Column j is contiguous
    x->data[j] /= Lj[j];

    // x[j+1:n] -= x[j] * L[j+1:n, j] using axpy with Vec wrappers
    size_t rem = n - j - 1;
    if (rem > 0) {
      Vec x_view = {.data = &x->data[j + 1], .rows = rem, .cols = 1};
      Vec col_view = {.data = (mat_elem_t *)&Lj[j + 1], .rows = rem, .cols = 1};
      mat_axpy(&x_view, -x->data[j], &col_view);
    }
  }
}
#endif

MATDEF void mat_solve_tril(Vec *x, const Mat *L, const Vec *b) {
  MAT_ASSERT_MAT(L);
  MAT_ASSERT_MAT(x);
  MAT_ASSERT_MAT(b);
  MAT_ASSERT_SQUARE(L);
  MAT_ASSERT(b->rows == L->rows);
  MAT_ASSERT(x->rows == L->rows);
#ifdef MAT_HAS_ARM_NEON
  mat_solve_tril_neon_(x, L, b);
#else
  mat_solve_tril_scalar_(x, L, b);
#endif
}

// --- mat_solve_tril_unit: Solve Lx = b, L unit lower triangular (1s on diag) -

MAT_INTERNAL_STATIC void mat_solve_tril_unit_scalar_(Vec *x, const Mat *L,
                                                      const Vec *b) {
  size_t n = L->rows;
  mat_elem_t *x_data = x->data;
  const mat_elem_t *b_data = b->data;

  // Forward substitution with unit diagonal: x[i] = b[i] - sum(L[i,j]*x[j], j<i)
  for (size_t i = 0; i < n; i++) {
    mat_elem_t dot = 0;
    for (size_t j = 0; j < i; j++) {
      dot += MAT_AT(L, i, j) * x_data[j];
    }
    x_data[i] = b_data[i] - dot;
  }
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void mat_solve_tril_unit_neon_(Vec *x, const Mat *L,
                                                    const Vec *b) {
  // Column-major: columns are contiguous, use column-oriented forward substitution
  // Unit diagonal: no division needed, just x[j+1:n] -= x[j] * L[j+1:n, j]
  size_t n = L->rows;
  const mat_elem_t *L_data = L->data;

  // Copy b to x (we'll modify x in place)
  memcpy(x->data, b->data, n * sizeof(mat_elem_t));

  for (size_t j = 0; j < n; j++) {
    const mat_elem_t *Lj = &L_data[j * n];  // Column j is contiguous

    // x[j+1:n] -= x[j] * L[j+1:n, j] using axpy with Vec wrappers
    size_t rem = n - j - 1;
    if (rem > 0) {
      Vec x_view = {.data = &x->data[j + 1], .rows = rem, .cols = 1};
      Vec col_view = {.data = (mat_elem_t *)&Lj[j + 1], .rows = rem, .cols = 1};
      mat_axpy(&x_view, -x->data[j], &col_view);
    }
  }
}
#endif

MATDEF void mat_solve_tril_unit(Vec *x, const Mat *L, const Vec *b) {
  MAT_ASSERT_MAT(L);
  MAT_ASSERT_MAT(x);
  MAT_ASSERT_MAT(b);
  MAT_ASSERT_SQUARE(L);
  MAT_ASSERT(b->rows == L->rows);
  MAT_ASSERT(x->rows == L->rows);
#ifdef MAT_HAS_ARM_NEON
  mat_solve_tril_unit_neon_(x, L, b);
#else
  mat_solve_tril_unit_scalar_(x, L, b);
#endif
}

// --- mat_solve_triu: Solve Ux = b, U upper triangular ---
// Row-oriented backward substitution with NEON optimization.
// For row-major U, accessing U[i, i+1:n] is contiguous.

#ifdef MAT_HAS_ARM_NEON
// Forward declaration for fallback
MAT_INTERNAL_STATIC void mat_solve_triu_scalar_(Vec *x, const Mat *U,
                                                 const Vec *b);

MAT_INTERNAL_STATIC void mat_solve_triu_neon_(Vec *x, const Mat *U,
                                               const Vec *b) {
  // Column-major: columns are contiguous, use column-oriented backward substitution
  // x[j] = b[j] / U[j,j], then update x[0:j] -= x[j] * U[0:j, j]
  size_t n = U->rows;
  const mat_elem_t *U_data = U->data;

  // Copy b to x (we'll modify x in place)
  memcpy(x->data, b->data, n * sizeof(mat_elem_t));

  for (size_t j = n; j-- > 0;) {
    const mat_elem_t *Uj = &U_data[j * n];  // Column j is contiguous
    x->data[j] /= Uj[j];

    // x[0:j] -= x[j] * U[0:j, j] using axpy with Vec wrappers
    if (j > 0) {
      Vec x_view = {.data = x->data, .rows = j, .cols = 1};
      Vec col_view = {.data = (mat_elem_t *)Uj, .rows = j, .cols = 1};
      mat_axpy(&x_view, -x->data[j], &col_view);
    }
  }
}
#endif // MAT_HAS_ARM_NEON

MAT_INTERNAL_STATIC void mat_solve_triu_scalar_(Vec *x, const Mat *U,
                                                 const Vec *b) {
  size_t n = U->rows;
  mat_elem_t *x_data = x->data;
  const mat_elem_t *b_data = b->data;

  // Back substitution: x[i] = (b[i] - sum(U[i,j]*x[j], j>i)) / U[i,i]
  for (size_t i = n; i-- > 0;) {
    mat_elem_t sum = b_data[i];
    for (size_t j = i + 1; j < n; j++) {
      sum -= MAT_AT(U, i, j) * x_data[j];
    }
    x_data[i] = sum / MAT_AT(U, i, i);
  }
}

MATDEF void mat_solve_triu(Vec *x, const Mat *U, const Vec *b) {
  MAT_ASSERT_MAT(U);
  MAT_ASSERT_MAT(x);
  MAT_ASSERT_MAT(b);
  MAT_ASSERT_SQUARE(U);
  MAT_ASSERT(b->rows == U->rows);
  MAT_ASSERT(x->rows == U->rows);
#ifdef MAT_HAS_ARM_NEON
  mat_solve_triu_neon_(x, U, b);
#else
  mat_solve_triu_scalar_(x, U, b);
#endif
}

// --- mat_solve_trilt: Solve L^T x = b, L lower triangular ---
// Uses column-oriented algorithm for contiguous memory access.

#ifdef MAT_HAS_ARM_NEON
// Forward declaration for fallback
MAT_INTERNAL_STATIC void mat_solve_trilt_scalar_(Vec *x, const Mat *L,
                                                  const Vec *b);

MAT_INTERNAL_STATIC void mat_solve_trilt_neon_(Vec *x, const Mat *L,
                                                const Vec *b) {
  // Column-major: columns are contiguous, use dot product approach
  // Solve L^T * x = b: x[j] = (b[j] - dot(L[j+1:n, j], x[j+1:n])) / L[j,j]
  size_t n = L->rows;
  mat_elem_t *x_data = x->data;
  const mat_elem_t *b_data = b->data;
  const mat_elem_t *L_data = L->data;

  for (size_t j = n; j-- > 0;) {
    const mat_elem_t *Lj = &L_data[j * n];  // Column j is contiguous
    mat_elem_t dot = mat_dot_raw_(&Lj[j + 1], &x_data[j + 1], n - j - 1);
    x_data[j] = (b_data[j] - dot) / Lj[j];
  }
}
#endif // MAT_HAS_ARM_NEON

MAT_INTERNAL_STATIC void mat_solve_trilt_scalar_(Vec *x, const Mat *L,
                                                  const Vec *b) {
  size_t n = L->rows;
  mat_elem_t *x_data = x->data;

  memcpy(x_data, b->data, n * sizeof(mat_elem_t));

  // Solve L^T * x = b using back substitution
  // L^T[i,j] = L[j,i], so L^T is upper triangular
  for (size_t j = n; j-- > 0;) {
    x_data[j] /= MAT_AT(L, j, j);
    mat_elem_t xj = x_data[j];
    for (size_t i = 0; i < j; i++) {
      // L^T[i,j] = L[j,i]
      x_data[i] -= MAT_AT(L, j, i) * xj;
    }
  }
}

MATDEF void mat_solve_trilt(Vec *x, const Mat *L, const Vec *b) {
  MAT_ASSERT_MAT(L);
  MAT_ASSERT_MAT(x);
  MAT_ASSERT_MAT(b);
  MAT_ASSERT_SQUARE(L);
  MAT_ASSERT(b->rows == L->rows);
  MAT_ASSERT(x->rows == L->rows);
#ifdef MAT_HAS_ARM_NEON
  mat_solve_trilt_neon_(x, L, b);
#else
  mat_solve_trilt_scalar_(x, L, b);
#endif
}

// ============================================================================
// Linear system solvers (using triangular solvers)
// ============================================================================

MATDEF void mat_solve(Vec *x, const Mat *A, const Vec *b) {
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(x);
  MAT_ASSERT_MAT(b);
  MAT_ASSERT_SQUARE(A);
  MAT_ASSERT(b->rows == A->rows && "b must have same rows as A");
  MAT_ASSERT(x->rows == A->cols && "x must have same rows as A cols");

  size_t n = A->rows;

  Mat *L = mat_mat(n, n);
  Mat *U = mat_mat(n, n);
  Perm *p = mat_perm(n);
  Vec *pb = mat_vec(n);
  Vec *z = mat_vec(n);

  mat_plu(A, L, U, p);

  for (size_t i = 0; i < n; i++) {
    pb->data[i] = b->data[p->data[i]];
  }

  mat_solve_tril_unit(z, L, pb);
  mat_solve_triu(x, U, z);

  MAT_FREE_MAT(L);
  MAT_FREE_MAT(U);
  MAT_FREE_PERM(p);
  MAT_FREE_MAT(pb);
  MAT_FREE_MAT(z);
}

MATDEF int mat_solve_spd(Vec *x, const Mat *A, const Vec *b) {
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(x);
  MAT_ASSERT_MAT(b);
  MAT_ASSERT_SQUARE(A);
  MAT_ASSERT(b->rows == A->rows && "b must have same rows as A");
  MAT_ASSERT(x->rows == A->cols && "x must have same rows as A cols");

  size_t n = A->rows;

  // Allocate temporaries
  Mat *L = mat_mat(n, n);
  Vec *y = mat_vec(n);

  // 1. Cholesky factorization: A = L * L^T
  int ret = mat_chol(A, L);
  if (ret != 0) {
    MAT_FREE_MAT(L);
    MAT_FREE_MAT(y);
    return -1; // Not positive definite
  }

  // 2. Forward substitution: Ly = b
  mat_solve_tril(y, L, b);

  // 3. Backward substitution: L^T x = y
  mat_solve_trilt(x, L, y);

  // Cleanup
  MAT_FREE_MAT(L);
  MAT_FREE_MAT(y);
  return 0;
}

// ============================================================================
// Cholesky decomposition (A = L * L^T, A must be symmetric positive definite)
// ============================================================================

#define MAT_CHOL_BLOCK_SIZE 64

// Lower-triangular GEMM: C[lower] += alpha * A * B^T[lower]
// Only computes and updates the lower triangle of C
// Uses GEMM-style 4x4 micro-kernel with packed B for cache efficiency
#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void
mat_gemm_lower_strided_neon_(mat_elem_t *C, size_t ldc, mat_elem_t alpha,
                             const mat_elem_t *A, size_t lda, size_t M,
                             size_t K, const mat_elem_t *B, size_t ldb) {
  // Pack B into contiguous memory (transpose: Bt[j,k] = B[j,k])
  // This gives sequential access in the k dimension
  mat_elem_t *Bt = (mat_elem_t *)mat_scratch_alloc_(M * K * sizeof(mat_elem_t));
  for (size_t j = 0; j < M; j++) {
    for (size_t k = 0; k < K; k++) {
      Bt[j * K + k] = B[j * ldb + k];
    }
  }

  // Process 4x4 blocks
  size_t i = 0;
  for (; i + 4 <= M; i += 4) {
    // Full 4x4 blocks strictly below diagonal (j + 4 <= i)
    size_t j = 0;
    for (; j + 4 <= i; j += 4) {
      MAT_NEON_TYPE acc00 = MAT_NEON_DUP(0), acc01 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc02 = MAT_NEON_DUP(0), acc03 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc10 = MAT_NEON_DUP(0), acc11 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc12 = MAT_NEON_DUP(0), acc13 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc20 = MAT_NEON_DUP(0), acc21 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc22 = MAT_NEON_DUP(0), acc23 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc30 = MAT_NEON_DUP(0), acc31 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc32 = MAT_NEON_DUP(0), acc33 = MAT_NEON_DUP(0);

      size_t k = 0;
      for (; k + MAT_NEON_WIDTH <= K; k += MAT_NEON_WIDTH) {
        MAT_NEON_TYPE a0 = MAT_NEON_LOAD(&A[(i + 0) * lda + k]);
        MAT_NEON_TYPE a1 = MAT_NEON_LOAD(&A[(i + 1) * lda + k]);
        MAT_NEON_TYPE a2 = MAT_NEON_LOAD(&A[(i + 2) * lda + k]);
        MAT_NEON_TYPE a3 = MAT_NEON_LOAD(&A[(i + 3) * lda + k]);
        MAT_NEON_TYPE b0 = MAT_NEON_LOAD(&Bt[(j + 0) * K + k]);
        MAT_NEON_TYPE b1 = MAT_NEON_LOAD(&Bt[(j + 1) * K + k]);
        MAT_NEON_TYPE b2 = MAT_NEON_LOAD(&Bt[(j + 2) * K + k]);
        MAT_NEON_TYPE b3 = MAT_NEON_LOAD(&Bt[(j + 3) * K + k]);
        acc00 = MAT_NEON_FMA(acc00, a0, b0);
        acc01 = MAT_NEON_FMA(acc01, a0, b1);
        acc02 = MAT_NEON_FMA(acc02, a0, b2);
        acc03 = MAT_NEON_FMA(acc03, a0, b3);
        acc10 = MAT_NEON_FMA(acc10, a1, b0);
        acc11 = MAT_NEON_FMA(acc11, a1, b1);
        acc12 = MAT_NEON_FMA(acc12, a1, b2);
        acc13 = MAT_NEON_FMA(acc13, a1, b3);
        acc20 = MAT_NEON_FMA(acc20, a2, b0);
        acc21 = MAT_NEON_FMA(acc21, a2, b1);
        acc22 = MAT_NEON_FMA(acc22, a2, b2);
        acc23 = MAT_NEON_FMA(acc23, a2, b3);
        acc30 = MAT_NEON_FMA(acc30, a3, b0);
        acc31 = MAT_NEON_FMA(acc31, a3, b1);
        acc32 = MAT_NEON_FMA(acc32, a3, b2);
        acc33 = MAT_NEON_FMA(acc33, a3, b3);
      }

      // Reduce and add to C with alpha scaling
      C[(i + 0) * ldc + (j + 0)] += alpha * MAT_NEON_ADDV(acc00);
      C[(i + 0) * ldc + (j + 1)] += alpha * MAT_NEON_ADDV(acc01);
      C[(i + 0) * ldc + (j + 2)] += alpha * MAT_NEON_ADDV(acc02);
      C[(i + 0) * ldc + (j + 3)] += alpha * MAT_NEON_ADDV(acc03);
      C[(i + 1) * ldc + (j + 0)] += alpha * MAT_NEON_ADDV(acc10);
      C[(i + 1) * ldc + (j + 1)] += alpha * MAT_NEON_ADDV(acc11);
      C[(i + 1) * ldc + (j + 2)] += alpha * MAT_NEON_ADDV(acc12);
      C[(i + 1) * ldc + (j + 3)] += alpha * MAT_NEON_ADDV(acc13);
      C[(i + 2) * ldc + (j + 0)] += alpha * MAT_NEON_ADDV(acc20);
      C[(i + 2) * ldc + (j + 1)] += alpha * MAT_NEON_ADDV(acc21);
      C[(i + 2) * ldc + (j + 2)] += alpha * MAT_NEON_ADDV(acc22);
      C[(i + 2) * ldc + (j + 3)] += alpha * MAT_NEON_ADDV(acc23);
      C[(i + 3) * ldc + (j + 0)] += alpha * MAT_NEON_ADDV(acc30);
      C[(i + 3) * ldc + (j + 1)] += alpha * MAT_NEON_ADDV(acc31);
      C[(i + 3) * ldc + (j + 2)] += alpha * MAT_NEON_ADDV(acc32);
      C[(i + 3) * ldc + (j + 3)] += alpha * MAT_NEON_ADDV(acc33);

      // Scalar remainder for k
      for (; k < K; k++) {
        mat_elem_t a0 = A[(i + 0) * lda + k], a1 = A[(i + 1) * lda + k];
        mat_elem_t a2 = A[(i + 2) * lda + k], a3 = A[(i + 3) * lda + k];
        mat_elem_t b0 = Bt[(j + 0) * K + k], b1 = Bt[(j + 1) * K + k];
        mat_elem_t b2 = Bt[(j + 2) * K + k], b3 = Bt[(j + 3) * K + k];
        C[(i + 0) * ldc + (j + 0)] += alpha * a0 * b0;
        C[(i + 0) * ldc + (j + 1)] += alpha * a0 * b1;
        C[(i + 0) * ldc + (j + 2)] += alpha * a0 * b2;
        C[(i + 0) * ldc + (j + 3)] += alpha * a0 * b3;
        C[(i + 1) * ldc + (j + 0)] += alpha * a1 * b0;
        C[(i + 1) * ldc + (j + 1)] += alpha * a1 * b1;
        C[(i + 1) * ldc + (j + 2)] += alpha * a1 * b2;
        C[(i + 1) * ldc + (j + 3)] += alpha * a1 * b3;
        C[(i + 2) * ldc + (j + 0)] += alpha * a2 * b0;
        C[(i + 2) * ldc + (j + 1)] += alpha * a2 * b1;
        C[(i + 2) * ldc + (j + 2)] += alpha * a2 * b2;
        C[(i + 2) * ldc + (j + 3)] += alpha * a2 * b3;
        C[(i + 3) * ldc + (j + 0)] += alpha * a3 * b0;
        C[(i + 3) * ldc + (j + 1)] += alpha * a3 * b1;
        C[(i + 3) * ldc + (j + 2)] += alpha * a3 * b2;
        C[(i + 3) * ldc + (j + 3)] += alpha * a3 * b3;
      }
    }

    // Remaining columns + diagonal block (lower triangle only)
    for (size_t ii = 0; ii < 4; ii++) {
      for (size_t jj = j; jj <= i + ii; jj++) {
        MAT_NEON_TYPE vsum = MAT_NEON_DUP(0);
        size_t k = 0;
        for (; k + MAT_NEON_WIDTH <= K; k += MAT_NEON_WIDTH) {
          MAT_NEON_TYPE va = MAT_NEON_LOAD(&A[(i + ii) * lda + k]);
          MAT_NEON_TYPE vb = MAT_NEON_LOAD(&Bt[jj * K + k]);
          vsum = MAT_NEON_FMA(vsum, va, vb);
        }
        mat_elem_t sum = MAT_NEON_ADDV(vsum);
        for (; k < K; k++) {
          sum += A[(i + ii) * lda + k] * Bt[jj * K + k];
        }
        C[(i + ii) * ldc + jj] += alpha * sum;
      }
    }
  }

  // Remaining rows
  for (; i < M; i++) {
    for (size_t j = 0; j <= i; j++) {
      MAT_NEON_TYPE vsum = MAT_NEON_DUP(0);
      size_t k = 0;
      for (; k + MAT_NEON_WIDTH <= K; k += MAT_NEON_WIDTH) {
        MAT_NEON_TYPE va = MAT_NEON_LOAD(&A[i * lda + k]);
        MAT_NEON_TYPE vb = MAT_NEON_LOAD(&Bt[j * K + k]);
        vsum = MAT_NEON_FMA(vsum, va, vb);
      }
      mat_elem_t sum = MAT_NEON_ADDV(vsum);
      for (; k < K; k++) {
        sum += A[i * lda + k] * Bt[j * K + k];
      }
      C[i * ldc + j] += alpha * sum;
    }
  }

#ifndef MAT_NO_SCRATCH
  mat_scratch_reset_();
#else
  mat_scratch_free_(Bt);
#endif
}
#endif

// Symmetric rank-k update for lower triangle: C = alpha * A * A^T + beta * C
// Scalar implementation
MAT_INTERNAL_STATIC void mat_syrk_lower_scalar_(mat_elem_t *C, size_t ldc,
                                                mat_elem_t alpha,
                                                const mat_elem_t *A, size_t lda,
                                                size_t m, size_t k,
                                                mat_elem_t beta) {
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j <= i; j++) {
      mat_elem_t sum = 0;
      for (size_t kk = 0; kk < k; kk++) {
        sum += A[i * lda + kk] * A[j * lda + kk];
      }
      C[i * ldc + j] = beta * C[i * ldc + j] + alpha * sum;
    }
  }
}

#ifdef MAT_HAS_ARM_NEON
// Symmetric rank-k update for lower triangle: C = alpha * A * A^T + beta * C
// NEON implementation with 4x4 blocking
MAT_INTERNAL_STATIC void mat_syrk_lower_neon_(mat_elem_t *C, size_t ldc,
                                              mat_elem_t alpha,
                                              const mat_elem_t *A, size_t lda,
                                              size_t m, size_t k,
                                              mat_elem_t beta) {
  // Process i rows in groups of 4
  size_t i = 0;
  for (; i + 4 <= m; i += 4) {
    // Off-diagonal: process j < i in 4x4 blocks
    size_t j = 0;
    for (; j + 4 <= i; j += 4) {
      // Accumulate dot products for 4x4 output block
      MAT_NEON_TYPE acc00 = MAT_NEON_DUP(0), acc01 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc02 = MAT_NEON_DUP(0), acc03 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc10 = MAT_NEON_DUP(0), acc11 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc12 = MAT_NEON_DUP(0), acc13 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc20 = MAT_NEON_DUP(0), acc21 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc22 = MAT_NEON_DUP(0), acc23 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc30 = MAT_NEON_DUP(0), acc31 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc32 = MAT_NEON_DUP(0), acc33 = MAT_NEON_DUP(0);

      size_t kk = 0;
      for (; kk + MAT_NEON_WIDTH <= k; kk += MAT_NEON_WIDTH) {
        MAT_NEON_TYPE a0 = MAT_NEON_LOAD(&A[(i + 0) * lda + kk]);
        MAT_NEON_TYPE a1 = MAT_NEON_LOAD(&A[(i + 1) * lda + kk]);
        MAT_NEON_TYPE a2 = MAT_NEON_LOAD(&A[(i + 2) * lda + kk]);
        MAT_NEON_TYPE a3 = MAT_NEON_LOAD(&A[(i + 3) * lda + kk]);
        MAT_NEON_TYPE b0 = MAT_NEON_LOAD(&A[(j + 0) * lda + kk]);
        MAT_NEON_TYPE b1 = MAT_NEON_LOAD(&A[(j + 1) * lda + kk]);
        MAT_NEON_TYPE b2 = MAT_NEON_LOAD(&A[(j + 2) * lda + kk]);
        MAT_NEON_TYPE b3 = MAT_NEON_LOAD(&A[(j + 3) * lda + kk]);
        acc00 = MAT_NEON_FMA(acc00, a0, b0);
        acc01 = MAT_NEON_FMA(acc01, a0, b1);
        acc02 = MAT_NEON_FMA(acc02, a0, b2);
        acc03 = MAT_NEON_FMA(acc03, a0, b3);
        acc10 = MAT_NEON_FMA(acc10, a1, b0);
        acc11 = MAT_NEON_FMA(acc11, a1, b1);
        acc12 = MAT_NEON_FMA(acc12, a1, b2);
        acc13 = MAT_NEON_FMA(acc13, a1, b3);
        acc20 = MAT_NEON_FMA(acc20, a2, b0);
        acc21 = MAT_NEON_FMA(acc21, a2, b1);
        acc22 = MAT_NEON_FMA(acc22, a2, b2);
        acc23 = MAT_NEON_FMA(acc23, a2, b3);
        acc30 = MAT_NEON_FMA(acc30, a3, b0);
        acc31 = MAT_NEON_FMA(acc31, a3, b1);
        acc32 = MAT_NEON_FMA(acc32, a3, b2);
        acc33 = MAT_NEON_FMA(acc33, a3, b3);
      }

      // Reduce NEON accumulators to scalars
      mat_elem_t s00 = MAT_NEON_ADDV(acc00), s01 = MAT_NEON_ADDV(acc01);
      mat_elem_t s02 = MAT_NEON_ADDV(acc02), s03 = MAT_NEON_ADDV(acc03);
      mat_elem_t s10 = MAT_NEON_ADDV(acc10), s11 = MAT_NEON_ADDV(acc11);
      mat_elem_t s12 = MAT_NEON_ADDV(acc12), s13 = MAT_NEON_ADDV(acc13);
      mat_elem_t s20 = MAT_NEON_ADDV(acc20), s21 = MAT_NEON_ADDV(acc21);
      mat_elem_t s22 = MAT_NEON_ADDV(acc22), s23 = MAT_NEON_ADDV(acc23);
      mat_elem_t s30 = MAT_NEON_ADDV(acc30), s31 = MAT_NEON_ADDV(acc31);
      mat_elem_t s32 = MAT_NEON_ADDV(acc32), s33 = MAT_NEON_ADDV(acc33);

      // Scalar tail for k
      for (; kk < k; kk++) {
        mat_elem_t a0 = A[(i + 0) * lda + kk], a1 = A[(i + 1) * lda + kk];
        mat_elem_t a2 = A[(i + 2) * lda + kk], a3 = A[(i + 3) * lda + kk];
        mat_elem_t b0 = A[(j + 0) * lda + kk], b1 = A[(j + 1) * lda + kk];
        mat_elem_t b2 = A[(j + 2) * lda + kk], b3 = A[(j + 3) * lda + kk];
        s00 += a0 * b0;
        s01 += a0 * b1;
        s02 += a0 * b2;
        s03 += a0 * b3;
        s10 += a1 * b0;
        s11 += a1 * b1;
        s12 += a1 * b2;
        s13 += a1 * b3;
        s20 += a2 * b0;
        s21 += a2 * b1;
        s22 += a2 * b2;
        s23 += a2 * b3;
        s30 += a3 * b0;
        s31 += a3 * b1;
        s32 += a3 * b2;
        s33 += a3 * b3;
      }

      // Store with alpha/beta
      C[(i + 0) * ldc + (j + 0)] =
          beta * C[(i + 0) * ldc + (j + 0)] + alpha * s00;
      C[(i + 0) * ldc + (j + 1)] =
          beta * C[(i + 0) * ldc + (j + 1)] + alpha * s01;
      C[(i + 0) * ldc + (j + 2)] =
          beta * C[(i + 0) * ldc + (j + 2)] + alpha * s02;
      C[(i + 0) * ldc + (j + 3)] =
          beta * C[(i + 0) * ldc + (j + 3)] + alpha * s03;
      C[(i + 1) * ldc + (j + 0)] =
          beta * C[(i + 1) * ldc + (j + 0)] + alpha * s10;
      C[(i + 1) * ldc + (j + 1)] =
          beta * C[(i + 1) * ldc + (j + 1)] + alpha * s11;
      C[(i + 1) * ldc + (j + 2)] =
          beta * C[(i + 1) * ldc + (j + 2)] + alpha * s12;
      C[(i + 1) * ldc + (j + 3)] =
          beta * C[(i + 1) * ldc + (j + 3)] + alpha * s13;
      C[(i + 2) * ldc + (j + 0)] =
          beta * C[(i + 2) * ldc + (j + 0)] + alpha * s20;
      C[(i + 2) * ldc + (j + 1)] =
          beta * C[(i + 2) * ldc + (j + 1)] + alpha * s21;
      C[(i + 2) * ldc + (j + 2)] =
          beta * C[(i + 2) * ldc + (j + 2)] + alpha * s22;
      C[(i + 2) * ldc + (j + 3)] =
          beta * C[(i + 2) * ldc + (j + 3)] + alpha * s23;
      C[(i + 3) * ldc + (j + 0)] =
          beta * C[(i + 3) * ldc + (j + 0)] + alpha * s30;
      C[(i + 3) * ldc + (j + 1)] =
          beta * C[(i + 3) * ldc + (j + 1)] + alpha * s31;
      C[(i + 3) * ldc + (j + 2)] =
          beta * C[(i + 3) * ldc + (j + 2)] + alpha * s32;
      C[(i + 3) * ldc + (j + 3)] =
          beta * C[(i + 3) * ldc + (j + 3)] + alpha * s33;
    }

    // Remaining columns before diagonal (j < i, one column at a time)
    for (; j < i; j++) {
      MAT_NEON_TYPE sum0 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE sum1 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE sum2 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE sum3 = MAT_NEON_DUP(0);
      size_t kk = 0;
      for (; kk + MAT_NEON_WIDTH <= k; kk += MAT_NEON_WIDTH) {
        MAT_NEON_TYPE vb = MAT_NEON_LOAD(&A[j * lda + kk]);
        sum0 = MAT_NEON_FMA(sum0, MAT_NEON_LOAD(&A[(i + 0) * lda + kk]), vb);
        sum1 = MAT_NEON_FMA(sum1, MAT_NEON_LOAD(&A[(i + 1) * lda + kk]), vb);
        sum2 = MAT_NEON_FMA(sum2, MAT_NEON_LOAD(&A[(i + 2) * lda + kk]), vb);
        sum3 = MAT_NEON_FMA(sum3, MAT_NEON_LOAD(&A[(i + 3) * lda + kk]), vb);
      }
      mat_elem_t s0 = MAT_NEON_ADDV(sum0);
      mat_elem_t s1 = MAT_NEON_ADDV(sum1);
      mat_elem_t s2 = MAT_NEON_ADDV(sum2);
      mat_elem_t s3 = MAT_NEON_ADDV(sum3);
      for (; kk < k; kk++) {
        mat_elem_t bk = A[j * lda + kk];
        s0 += A[(i + 0) * lda + kk] * bk;
        s1 += A[(i + 1) * lda + kk] * bk;
        s2 += A[(i + 2) * lda + kk] * bk;
        s3 += A[(i + 3) * lda + kk] * bk;
      }
      C[(i + 0) * ldc + j] = beta * C[(i + 0) * ldc + j] + alpha * s0;
      C[(i + 1) * ldc + j] = beta * C[(i + 1) * ldc + j] + alpha * s1;
      C[(i + 2) * ldc + j] = beta * C[(i + 2) * ldc + j] + alpha * s2;
      C[(i + 3) * ldc + j] = beta * C[(i + 3) * ldc + j] + alpha * s3;
    }

    // Diagonal block: elements where i <= row < i+4 and i <= col <= row
    for (size_t ii = 0; ii < 4; ii++) {
      for (size_t jj = 0; jj <= ii; jj++) {
        MAT_NEON_TYPE vsum = MAT_NEON_DUP(0);
        size_t kk = 0;
        for (; kk + MAT_NEON_WIDTH <= k; kk += MAT_NEON_WIDTH) {
          MAT_NEON_TYPE va = MAT_NEON_LOAD(&A[(i + ii) * lda + kk]);
          MAT_NEON_TYPE vb = MAT_NEON_LOAD(&A[(i + jj) * lda + kk]);
          vsum = MAT_NEON_FMA(vsum, va, vb);
        }
        mat_elem_t sum = MAT_NEON_ADDV(vsum);
        for (; kk < k; kk++) {
          sum += A[(i + ii) * lda + kk] * A[(i + jj) * lda + kk];
        }
        C[(i + ii) * ldc + (i + jj)] =
            beta * C[(i + ii) * ldc + (i + jj)] + alpha * sum;
      }
    }
  }

  // Remaining rows (less than 4)
  for (; i < m; i++) {
    for (size_t j = 0; j <= i; j++) {
      MAT_NEON_TYPE vsum = MAT_NEON_DUP(0);
      size_t kk = 0;
      for (; kk + MAT_NEON_WIDTH <= k; kk += MAT_NEON_WIDTH) {
        MAT_NEON_TYPE va = MAT_NEON_LOAD(&A[i * lda + kk]);
        MAT_NEON_TYPE vb = MAT_NEON_LOAD(&A[j * lda + kk]);
        vsum = MAT_NEON_FMA(vsum, va, vb);
      }
      mat_elem_t sum = MAT_NEON_ADDV(vsum);
      for (; kk < k; kk++) {
        sum += A[i * lda + kk] * A[j * lda + kk];
      }
      C[i * ldc + j] = beta * C[i * ldc + j] + alpha * sum;
    }
  }
}
#endif

// Symmetric rank-k update for lower triangle: C = alpha * A * A^T + beta * C
MAT_INTERNAL_STATIC void mat_syrk_lower_(mat_elem_t *C, size_t ldc,
                                         mat_elem_t alpha, const mat_elem_t *A,
                                         size_t lda, size_t m, size_t k,
                                         mat_elem_t beta) {
#ifdef MAT_HAS_ARM_NEON
  mat_syrk_lower_neon_(C, ldc, alpha, A, lda, m, k, beta);
#else
  mat_syrk_lower_scalar_(C, ldc, alpha, A, lda, m, k, beta);
#endif
}

// Symmetric rank-k update for upper triangle: C = alpha * A * A^T + beta * C
// Scalar implementation
MAT_INTERNAL_STATIC void mat_syrk_upper_scalar_(mat_elem_t *C, size_t ldc,
                                                mat_elem_t alpha,
                                                const mat_elem_t *A, size_t lda,
                                                size_t m, size_t k,
                                                mat_elem_t beta) {
  for (size_t i = 0; i < m; i++) {
    for (size_t j = i; j < m; j++) {
      mat_elem_t sum = 0;
      for (size_t kk = 0; kk < k; kk++) {
        sum += A[i * lda + kk] * A[j * lda + kk];
      }
      C[i * ldc + j] = beta * C[i * ldc + j] + alpha * sum;
    }
  }
}

#ifdef MAT_HAS_ARM_NEON
// Symmetric rank-k update for upper triangle: C = alpha * A * A^T + beta * C
// NEON implementation with 4x4 blocking
MAT_INTERNAL_STATIC void mat_syrk_upper_neon_(mat_elem_t *C, size_t ldc,
                                              mat_elem_t alpha,
                                              const mat_elem_t *A, size_t lda,
                                              size_t m, size_t k,
                                              mat_elem_t beta) {
  // Process i rows in groups of 4
  size_t i = 0;
  for (; i + 4 <= m; i += 4) {
    // Diagonal block first: elements where i <= row < i+4 and row <= col < i+4
    for (size_t ii = 0; ii < 4; ii++) {
      for (size_t jj = ii; jj < 4; jj++) {
        MAT_NEON_TYPE vsum = MAT_NEON_DUP(0);
        size_t kk = 0;
        for (; kk + MAT_NEON_WIDTH <= k; kk += MAT_NEON_WIDTH) {
          MAT_NEON_TYPE va = MAT_NEON_LOAD(&A[(i + ii) * lda + kk]);
          MAT_NEON_TYPE vb = MAT_NEON_LOAD(&A[(i + jj) * lda + kk]);
          vsum = MAT_NEON_FMA(vsum, va, vb);
        }
        mat_elem_t sum = MAT_NEON_ADDV(vsum);
        for (; kk < k; kk++) {
          sum += A[(i + ii) * lda + kk] * A[(i + jj) * lda + kk];
        }
        C[(i + ii) * ldc + (i + jj)] =
            beta * C[(i + ii) * ldc + (i + jj)] + alpha * sum;
      }
    }

    // Off-diagonal: process j > i+3 in 4x4 blocks
    size_t j = i + 4;
    for (; j + 4 <= m; j += 4) {
      // Accumulate dot products for 4x4 output block
      MAT_NEON_TYPE acc00 = MAT_NEON_DUP(0), acc01 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc02 = MAT_NEON_DUP(0), acc03 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc10 = MAT_NEON_DUP(0), acc11 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc12 = MAT_NEON_DUP(0), acc13 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc20 = MAT_NEON_DUP(0), acc21 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc22 = MAT_NEON_DUP(0), acc23 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc30 = MAT_NEON_DUP(0), acc31 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc32 = MAT_NEON_DUP(0), acc33 = MAT_NEON_DUP(0);

      size_t kk = 0;
      for (; kk + MAT_NEON_WIDTH <= k; kk += MAT_NEON_WIDTH) {
        MAT_NEON_TYPE a0 = MAT_NEON_LOAD(&A[(i + 0) * lda + kk]);
        MAT_NEON_TYPE a1 = MAT_NEON_LOAD(&A[(i + 1) * lda + kk]);
        MAT_NEON_TYPE a2 = MAT_NEON_LOAD(&A[(i + 2) * lda + kk]);
        MAT_NEON_TYPE a3 = MAT_NEON_LOAD(&A[(i + 3) * lda + kk]);
        MAT_NEON_TYPE b0 = MAT_NEON_LOAD(&A[(j + 0) * lda + kk]);
        MAT_NEON_TYPE b1 = MAT_NEON_LOAD(&A[(j + 1) * lda + kk]);
        MAT_NEON_TYPE b2 = MAT_NEON_LOAD(&A[(j + 2) * lda + kk]);
        MAT_NEON_TYPE b3 = MAT_NEON_LOAD(&A[(j + 3) * lda + kk]);
        acc00 = MAT_NEON_FMA(acc00, a0, b0);
        acc01 = MAT_NEON_FMA(acc01, a0, b1);
        acc02 = MAT_NEON_FMA(acc02, a0, b2);
        acc03 = MAT_NEON_FMA(acc03, a0, b3);
        acc10 = MAT_NEON_FMA(acc10, a1, b0);
        acc11 = MAT_NEON_FMA(acc11, a1, b1);
        acc12 = MAT_NEON_FMA(acc12, a1, b2);
        acc13 = MAT_NEON_FMA(acc13, a1, b3);
        acc20 = MAT_NEON_FMA(acc20, a2, b0);
        acc21 = MAT_NEON_FMA(acc21, a2, b1);
        acc22 = MAT_NEON_FMA(acc22, a2, b2);
        acc23 = MAT_NEON_FMA(acc23, a2, b3);
        acc30 = MAT_NEON_FMA(acc30, a3, b0);
        acc31 = MAT_NEON_FMA(acc31, a3, b1);
        acc32 = MAT_NEON_FMA(acc32, a3, b2);
        acc33 = MAT_NEON_FMA(acc33, a3, b3);
      }

      // Reduce NEON accumulators to scalars
      mat_elem_t s00 = MAT_NEON_ADDV(acc00), s01 = MAT_NEON_ADDV(acc01);
      mat_elem_t s02 = MAT_NEON_ADDV(acc02), s03 = MAT_NEON_ADDV(acc03);
      mat_elem_t s10 = MAT_NEON_ADDV(acc10), s11 = MAT_NEON_ADDV(acc11);
      mat_elem_t s12 = MAT_NEON_ADDV(acc12), s13 = MAT_NEON_ADDV(acc13);
      mat_elem_t s20 = MAT_NEON_ADDV(acc20), s21 = MAT_NEON_ADDV(acc21);
      mat_elem_t s22 = MAT_NEON_ADDV(acc22), s23 = MAT_NEON_ADDV(acc23);
      mat_elem_t s30 = MAT_NEON_ADDV(acc30), s31 = MAT_NEON_ADDV(acc31);
      mat_elem_t s32 = MAT_NEON_ADDV(acc32), s33 = MAT_NEON_ADDV(acc33);

      // Scalar tail for k
      for (; kk < k; kk++) {
        mat_elem_t a0 = A[(i + 0) * lda + kk], a1 = A[(i + 1) * lda + kk];
        mat_elem_t a2 = A[(i + 2) * lda + kk], a3 = A[(i + 3) * lda + kk];
        mat_elem_t b0 = A[(j + 0) * lda + kk], b1 = A[(j + 1) * lda + kk];
        mat_elem_t b2 = A[(j + 2) * lda + kk], b3 = A[(j + 3) * lda + kk];
        s00 += a0 * b0;
        s01 += a0 * b1;
        s02 += a0 * b2;
        s03 += a0 * b3;
        s10 += a1 * b0;
        s11 += a1 * b1;
        s12 += a1 * b2;
        s13 += a1 * b3;
        s20 += a2 * b0;
        s21 += a2 * b1;
        s22 += a2 * b2;
        s23 += a2 * b3;
        s30 += a3 * b0;
        s31 += a3 * b1;
        s32 += a3 * b2;
        s33 += a3 * b3;
      }

      // Store with alpha/beta
      C[(i + 0) * ldc + (j + 0)] =
          beta * C[(i + 0) * ldc + (j + 0)] + alpha * s00;
      C[(i + 0) * ldc + (j + 1)] =
          beta * C[(i + 0) * ldc + (j + 1)] + alpha * s01;
      C[(i + 0) * ldc + (j + 2)] =
          beta * C[(i + 0) * ldc + (j + 2)] + alpha * s02;
      C[(i + 0) * ldc + (j + 3)] =
          beta * C[(i + 0) * ldc + (j + 3)] + alpha * s03;
      C[(i + 1) * ldc + (j + 0)] =
          beta * C[(i + 1) * ldc + (j + 0)] + alpha * s10;
      C[(i + 1) * ldc + (j + 1)] =
          beta * C[(i + 1) * ldc + (j + 1)] + alpha * s11;
      C[(i + 1) * ldc + (j + 2)] =
          beta * C[(i + 1) * ldc + (j + 2)] + alpha * s12;
      C[(i + 1) * ldc + (j + 3)] =
          beta * C[(i + 1) * ldc + (j + 3)] + alpha * s13;
      C[(i + 2) * ldc + (j + 0)] =
          beta * C[(i + 2) * ldc + (j + 0)] + alpha * s20;
      C[(i + 2) * ldc + (j + 1)] =
          beta * C[(i + 2) * ldc + (j + 1)] + alpha * s21;
      C[(i + 2) * ldc + (j + 2)] =
          beta * C[(i + 2) * ldc + (j + 2)] + alpha * s22;
      C[(i + 2) * ldc + (j + 3)] =
          beta * C[(i + 2) * ldc + (j + 3)] + alpha * s23;
      C[(i + 3) * ldc + (j + 0)] =
          beta * C[(i + 3) * ldc + (j + 0)] + alpha * s30;
      C[(i + 3) * ldc + (j + 1)] =
          beta * C[(i + 3) * ldc + (j + 1)] + alpha * s31;
      C[(i + 3) * ldc + (j + 2)] =
          beta * C[(i + 3) * ldc + (j + 2)] + alpha * s32;
      C[(i + 3) * ldc + (j + 3)] =
          beta * C[(i + 3) * ldc + (j + 3)] + alpha * s33;
    }

    // Remaining columns after diagonal (j > i+3, one column at a time)
    for (; j < m; j++) {
      MAT_NEON_TYPE sum0 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE sum1 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE sum2 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE sum3 = MAT_NEON_DUP(0);
      size_t kk = 0;
      for (; kk + MAT_NEON_WIDTH <= k; kk += MAT_NEON_WIDTH) {
        MAT_NEON_TYPE vb = MAT_NEON_LOAD(&A[j * lda + kk]);
        sum0 = MAT_NEON_FMA(sum0, MAT_NEON_LOAD(&A[(i + 0) * lda + kk]), vb);
        sum1 = MAT_NEON_FMA(sum1, MAT_NEON_LOAD(&A[(i + 1) * lda + kk]), vb);
        sum2 = MAT_NEON_FMA(sum2, MAT_NEON_LOAD(&A[(i + 2) * lda + kk]), vb);
        sum3 = MAT_NEON_FMA(sum3, MAT_NEON_LOAD(&A[(i + 3) * lda + kk]), vb);
      }
      mat_elem_t s0 = MAT_NEON_ADDV(sum0);
      mat_elem_t s1 = MAT_NEON_ADDV(sum1);
      mat_elem_t s2 = MAT_NEON_ADDV(sum2);
      mat_elem_t s3 = MAT_NEON_ADDV(sum3);
      for (; kk < k; kk++) {
        mat_elem_t bk = A[j * lda + kk];
        s0 += A[(i + 0) * lda + kk] * bk;
        s1 += A[(i + 1) * lda + kk] * bk;
        s2 += A[(i + 2) * lda + kk] * bk;
        s3 += A[(i + 3) * lda + kk] * bk;
      }
      C[(i + 0) * ldc + j] = beta * C[(i + 0) * ldc + j] + alpha * s0;
      C[(i + 1) * ldc + j] = beta * C[(i + 1) * ldc + j] + alpha * s1;
      C[(i + 2) * ldc + j] = beta * C[(i + 2) * ldc + j] + alpha * s2;
      C[(i + 3) * ldc + j] = beta * C[(i + 3) * ldc + j] + alpha * s3;
    }
  }

  // Remaining rows (less than 4)
  for (; i < m; i++) {
    for (size_t j = i; j < m; j++) {
      MAT_NEON_TYPE vsum = MAT_NEON_DUP(0);
      size_t kk = 0;
      for (; kk + MAT_NEON_WIDTH <= k; kk += MAT_NEON_WIDTH) {
        MAT_NEON_TYPE va = MAT_NEON_LOAD(&A[i * lda + kk]);
        MAT_NEON_TYPE vb = MAT_NEON_LOAD(&A[j * lda + kk]);
        vsum = MAT_NEON_FMA(vsum, va, vb);
      }
      mat_elem_t sum = MAT_NEON_ADDV(vsum);
      for (; kk < k; kk++) {
        sum += A[i * lda + kk] * A[j * lda + kk];
      }
      C[i * ldc + j] = beta * C[i * ldc + j] + alpha * sum;
    }
  }
}
#endif

// Symmetric rank-k update for upper triangle: C = alpha * A * A^T + beta * C
MAT_INTERNAL_STATIC void mat_syrk_upper_(mat_elem_t *C, size_t ldc,
                                         mat_elem_t alpha, const mat_elem_t *A,
                                         size_t lda, size_t m, size_t k,
                                         mat_elem_t beta) {
#ifdef MAT_HAS_ARM_NEON
  mat_syrk_upper_neon_(C, ldc, alpha, A, lda, m, k, beta);
#else
  mat_syrk_upper_scalar_(C, ldc, alpha, A, lda, m, k, beta);
#endif
}

// ============================================================================
// Column-major SYRK: C = alpha * A * A^T + beta * C
// A is m×k column-major: A[i,p] = A[p*lda + i] (column p contiguous)
// C is m×m column-major: C[i,j] = C[j*ldc + i] (column j contiguous)
// Uses rank-1 update algorithm for cache-friendly column access
// ============================================================================

// Column-major SYRK lower triangle - scalar implementation
MAT_INTERNAL_STATIC void mat_syrk_lower_colmajor_scalar_(
    mat_elem_t *C, size_t ldc, mat_elem_t alpha, const mat_elem_t *A,
    size_t lda, size_t m, size_t k, mat_elem_t beta) {
  // Scale C by beta (lower triangle only)
  if (beta == 0) {
    for (size_t j = 0; j < m; j++)
      for (size_t i = j; i < m; i++)
        C[j * ldc + i] = 0;
  } else if (beta != 1) {
    for (size_t j = 0; j < m; j++)
      for (size_t i = j; i < m; i++)
        C[j * ldc + i] *= beta;
  }

  // Rank-1 updates: C += alpha * A[:,p] * A[:,p]^T
  for (size_t p = 0; p < k; p++) {
    const mat_elem_t *Ap = &A[p * lda];  // Column p of A (contiguous)
    for (size_t j = 0; j < m; j++) {
      mat_elem_t scalar = alpha * Ap[j];
      mat_elem_t *Cj = &C[j * ldc];  // Column j of C (contiguous)
      for (size_t i = j; i < m; i++) {
        Cj[i] += scalar * Ap[i];
      }
    }
  }
}

#ifdef MAT_HAS_ARM_NEON
// Column-major SYRK lower triangle - simple version for small problems
MAT_INTERNAL_STATIC void mat_syrk_lower_colmajor_unblocked_(
    mat_elem_t *C, size_t ldc, mat_elem_t alpha, const mat_elem_t *A,
    size_t lda, size_t m, size_t k, mat_elem_t beta) {
  const size_t NW = MAT_NEON_WIDTH;
  MAT_NEON_TYPE valpha = MAT_NEON_DUP(alpha);
  MAT_NEON_TYPE vbeta = MAT_NEON_DUP(beta);

  size_t j = 0;
  for (; j + NW <= m; j += NW) {
    // Off-diagonal 4x4 blocks
    size_t i = j + NW;
    for (; i + NW <= m; i += NW) {
      MAT_NEON_TYPE acc0 = MAT_NEON_DUP(0), acc1 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc2 = MAT_NEON_DUP(0), acc3 = MAT_NEON_DUP(0);

      size_t p = 0;
      for (; p + 4 <= k; p += 4) {
        MAT_NEON_TYPE b0 = MAT_NEON_LOAD(&A[(p + 0) * lda + j]);
        MAT_NEON_TYPE b1 = MAT_NEON_LOAD(&A[(p + 1) * lda + j]);
        MAT_NEON_TYPE b2 = MAT_NEON_LOAD(&A[(p + 2) * lda + j]);
        MAT_NEON_TYPE b3 = MAT_NEON_LOAD(&A[(p + 3) * lda + j]);
        MAT_NEON_TYPE a0 = MAT_NEON_LOAD(&A[(p + 0) * lda + i]);
        MAT_NEON_TYPE a1 = MAT_NEON_LOAD(&A[(p + 1) * lda + i]);
        MAT_NEON_TYPE a2 = MAT_NEON_LOAD(&A[(p + 2) * lda + i]);
        MAT_NEON_TYPE a3 = MAT_NEON_LOAD(&A[(p + 3) * lda + i]);
        acc0 = MAT_NEON_FMA_LANE(acc0, a0, b0, 0);
        acc0 = MAT_NEON_FMA_LANE(acc0, a1, b1, 0);
        acc0 = MAT_NEON_FMA_LANE(acc0, a2, b2, 0);
        acc0 = MAT_NEON_FMA_LANE(acc0, a3, b3, 0);
        acc1 = MAT_NEON_FMA_LANE(acc1, a0, b0, 1);
        acc1 = MAT_NEON_FMA_LANE(acc1, a1, b1, 1);
        acc1 = MAT_NEON_FMA_LANE(acc1, a2, b2, 1);
        acc1 = MAT_NEON_FMA_LANE(acc1, a3, b3, 1);
        acc2 = MAT_NEON_FMA_LANE(acc2, a0, b0, 2);
        acc2 = MAT_NEON_FMA_LANE(acc2, a1, b1, 2);
        acc2 = MAT_NEON_FMA_LANE(acc2, a2, b2, 2);
        acc2 = MAT_NEON_FMA_LANE(acc2, a3, b3, 2);
        acc3 = MAT_NEON_FMA_LANE(acc3, a0, b0, 3);
        acc3 = MAT_NEON_FMA_LANE(acc3, a1, b1, 3);
        acc3 = MAT_NEON_FMA_LANE(acc3, a2, b2, 3);
        acc3 = MAT_NEON_FMA_LANE(acc3, a3, b3, 3);
      }
      for (; p < k; p++) {
        MAT_NEON_TYPE a = MAT_NEON_LOAD(&A[p * lda + i]);
        MAT_NEON_TYPE b = MAT_NEON_LOAD(&A[p * lda + j]);
        acc0 = MAT_NEON_FMA_LANE(acc0, a, b, 0);
        acc1 = MAT_NEON_FMA_LANE(acc1, a, b, 1);
        acc2 = MAT_NEON_FMA_LANE(acc2, a, b, 2);
        acc3 = MAT_NEON_FMA_LANE(acc3, a, b, 3);
      }

      mat_elem_t *C0 = &C[(j + 0) * ldc + i];
      mat_elem_t *C1 = &C[(j + 1) * ldc + i];
      mat_elem_t *C2 = &C[(j + 2) * ldc + i];
      mat_elem_t *C3 = &C[(j + 3) * ldc + i];
      if (beta == 0) {
        MAT_NEON_STORE(C0, MAT_NEON_MUL(valpha, acc0));
        MAT_NEON_STORE(C1, MAT_NEON_MUL(valpha, acc1));
        MAT_NEON_STORE(C2, MAT_NEON_MUL(valpha, acc2));
        MAT_NEON_STORE(C3, MAT_NEON_MUL(valpha, acc3));
      } else {
        MAT_NEON_STORE(C0, MAT_NEON_FMA(MAT_NEON_MUL(vbeta, MAT_NEON_LOAD(C0)),
                                        valpha, acc0));
        MAT_NEON_STORE(C1, MAT_NEON_FMA(MAT_NEON_MUL(vbeta, MAT_NEON_LOAD(C1)),
                                        valpha, acc1));
        MAT_NEON_STORE(C2, MAT_NEON_FMA(MAT_NEON_MUL(vbeta, MAT_NEON_LOAD(C2)),
                                        valpha, acc2));
        MAT_NEON_STORE(C3, MAT_NEON_FMA(MAT_NEON_MUL(vbeta, MAT_NEON_LOAD(C3)),
                                        valpha, acc3));
      }
    }

    // Remaining rows
    for (; i < m; i++) {
      for (size_t jj = j; jj < j + NW; jj++) {
        mat_elem_t sum = 0;
        for (size_t p = 0; p < k; p++)
          sum += A[p * lda + i] * A[p * lda + jj];
        C[jj * ldc + i] = beta * C[jj * ldc + i] + alpha * sum;
      }
    }

    // Diagonal block
    {
      MAT_NEON_TYPE acc0 = MAT_NEON_DUP(0), acc1 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc2 = MAT_NEON_DUP(0), acc3 = MAT_NEON_DUP(0);
      for (size_t p = 0; p + 4 <= k; p += 4) {
        MAT_NEON_TYPE b0 = MAT_NEON_LOAD(&A[(p + 0) * lda + j]);
        MAT_NEON_TYPE b1 = MAT_NEON_LOAD(&A[(p + 1) * lda + j]);
        MAT_NEON_TYPE b2 = MAT_NEON_LOAD(&A[(p + 2) * lda + j]);
        MAT_NEON_TYPE b3 = MAT_NEON_LOAD(&A[(p + 3) * lda + j]);
        acc0 = MAT_NEON_FMA_LANE(acc0, b0, b0, 0);
        acc0 = MAT_NEON_FMA_LANE(acc0, b1, b1, 0);
        acc0 = MAT_NEON_FMA_LANE(acc0, b2, b2, 0);
        acc0 = MAT_NEON_FMA_LANE(acc0, b3, b3, 0);
        acc1 = MAT_NEON_FMA_LANE(acc1, b0, b0, 1);
        acc1 = MAT_NEON_FMA_LANE(acc1, b1, b1, 1);
        acc1 = MAT_NEON_FMA_LANE(acc1, b2, b2, 1);
        acc1 = MAT_NEON_FMA_LANE(acc1, b3, b3, 1);
        acc2 = MAT_NEON_FMA_LANE(acc2, b0, b0, 2);
        acc2 = MAT_NEON_FMA_LANE(acc2, b1, b1, 2);
        acc2 = MAT_NEON_FMA_LANE(acc2, b2, b2, 2);
        acc2 = MAT_NEON_FMA_LANE(acc2, b3, b3, 2);
        acc3 = MAT_NEON_FMA_LANE(acc3, b0, b0, 3);
        acc3 = MAT_NEON_FMA_LANE(acc3, b1, b1, 3);
        acc3 = MAT_NEON_FMA_LANE(acc3, b2, b2, 3);
        acc3 = MAT_NEON_FMA_LANE(acc3, b3, b3, 3);
      }
      for (size_t p = (k / 4) * 4; p < k; p++) {
        MAT_NEON_TYPE b = MAT_NEON_LOAD(&A[p * lda + j]);
        acc0 = MAT_NEON_FMA_LANE(acc0, b, b, 0);
        acc1 = MAT_NEON_FMA_LANE(acc1, b, b, 1);
        acc2 = MAT_NEON_FMA_LANE(acc2, b, b, 2);
        acc3 = MAT_NEON_FMA_LANE(acc3, b, b, 3);
      }
      mat_elem_t s[4][4];
      MAT_NEON_STORE(s[0], acc0);
      MAT_NEON_STORE(s[1], acc1);
      MAT_NEON_STORE(s[2], acc2);
      MAT_NEON_STORE(s[3], acc3);
      for (size_t col = 0; col < NW; col++) {
        for (size_t row = col; row < NW; row++) {
          mat_elem_t *Cptr = &C[(j + col) * ldc + (j + row)];
          *Cptr = beta * (*Cptr) + alpha * s[col][row];
        }
      }
    }
  }

  // Remaining columns
  for (; j < m; j++) {
    for (size_t i = j; i < m; i++) {
      mat_elem_t sum = 0;
      for (size_t p = 0; p < k; p++)
        sum += A[p * lda + i] * A[p * lda + j];
      C[j * ldc + i] = beta * C[j * ldc + i] + alpha * sum;
    }
  }
}

// Column-major SYRK lower triangle - k-blocked version for larger problems
#define MAT_SYRK_K_BLOCK 32
MAT_INTERNAL_STATIC void mat_syrk_lower_colmajor_blocked_(
    mat_elem_t *C, size_t ldc, mat_elem_t alpha, const mat_elem_t *A,
    size_t lda, size_t m, size_t k, mat_elem_t beta) {
  const size_t NW = MAT_NEON_WIDTH;

  // Scale C by beta first (only once)
  if (beta == 0) {
    for (size_t jj = 0; jj < m; jj++) {
      mat_elem_t *Cj = &C[jj * ldc + jj];
      size_t len = m - jj;
      size_t ii = 0;
      for (; ii + NW * 4 <= len; ii += NW * 4) {
        MAT_NEON_STORE(&Cj[ii], MAT_NEON_DUP(0));
        MAT_NEON_STORE(&Cj[ii + NW], MAT_NEON_DUP(0));
        MAT_NEON_STORE(&Cj[ii + NW * 2], MAT_NEON_DUP(0));
        MAT_NEON_STORE(&Cj[ii + NW * 3], MAT_NEON_DUP(0));
      }
      for (; ii < len; ii++)
        Cj[ii] = 0;
    }
  } else if (beta != 1) {
    MAT_NEON_TYPE vbeta = MAT_NEON_DUP(beta);
    for (size_t jj = 0; jj < m; jj++) {
      mat_elem_t *Cj = &C[jj * ldc + jj];
      size_t len = m - jj;
      size_t ii = 0;
      for (; ii + NW * 4 <= len; ii += NW * 4) {
        MAT_NEON_STORE(&Cj[ii], MAT_NEON_MUL(vbeta, MAT_NEON_LOAD(&Cj[ii])));
        MAT_NEON_STORE(&Cj[ii + NW],
                       MAT_NEON_MUL(vbeta, MAT_NEON_LOAD(&Cj[ii + NW])));
        MAT_NEON_STORE(&Cj[ii + NW * 2],
                       MAT_NEON_MUL(vbeta, MAT_NEON_LOAD(&Cj[ii + NW * 2])));
        MAT_NEON_STORE(&Cj[ii + NW * 3],
                       MAT_NEON_MUL(vbeta, MAT_NEON_LOAD(&Cj[ii + NW * 3])));
      }
      for (; ii < len; ii++)
        Cj[ii] *= beta;
    }
  }

  MAT_NEON_TYPE valpha = MAT_NEON_DUP(alpha);

  // Process k in blocks for better cache reuse
  for (size_t kb = 0; kb < k; kb += MAT_SYRK_K_BLOCK) {
    size_t k_end = (kb + MAT_SYRK_K_BLOCK < k) ? kb + MAT_SYRK_K_BLOCK : k;

    // Process 4-column blocks of C
    size_t j = 0;
    for (; j + NW <= m; j += NW) {
      // Off-diagonal 8x4 blocks
      size_t i = j + NW;
      for (; i + NW * 2 <= m; i += NW * 2) {
        MAT_NEON_TYPE acc00 = MAT_NEON_DUP(0), acc01 = MAT_NEON_DUP(0);
        MAT_NEON_TYPE acc10 = MAT_NEON_DUP(0), acc11 = MAT_NEON_DUP(0);
        MAT_NEON_TYPE acc20 = MAT_NEON_DUP(0), acc21 = MAT_NEON_DUP(0);
        MAT_NEON_TYPE acc30 = MAT_NEON_DUP(0), acc31 = MAT_NEON_DUP(0);

        size_t p = kb;
        for (; p + 4 <= k_end; p += 4) {
          // Prefetch next columns
          __builtin_prefetch(&A[(p + 4) * lda + i], 0, 3);
          __builtin_prefetch(&A[(p + 4) * lda + j], 0, 3);

          MAT_NEON_TYPE b0 = MAT_NEON_LOAD(&A[(p + 0) * lda + j]);
          MAT_NEON_TYPE b1 = MAT_NEON_LOAD(&A[(p + 1) * lda + j]);
          MAT_NEON_TYPE b2 = MAT_NEON_LOAD(&A[(p + 2) * lda + j]);
          MAT_NEON_TYPE b3 = MAT_NEON_LOAD(&A[(p + 3) * lda + j]);

          MAT_NEON_TYPE a00 = MAT_NEON_LOAD(&A[(p + 0) * lda + i]);
          MAT_NEON_TYPE a01 = MAT_NEON_LOAD(&A[(p + 0) * lda + i + NW]);
          acc00 = MAT_NEON_FMA_LANE(acc00, a00, b0, 0);
          acc01 = MAT_NEON_FMA_LANE(acc01, a01, b0, 0);
          acc10 = MAT_NEON_FMA_LANE(acc10, a00, b0, 1);
          acc11 = MAT_NEON_FMA_LANE(acc11, a01, b0, 1);
          acc20 = MAT_NEON_FMA_LANE(acc20, a00, b0, 2);
          acc21 = MAT_NEON_FMA_LANE(acc21, a01, b0, 2);
          acc30 = MAT_NEON_FMA_LANE(acc30, a00, b0, 3);
          acc31 = MAT_NEON_FMA_LANE(acc31, a01, b0, 3);

          MAT_NEON_TYPE a10 = MAT_NEON_LOAD(&A[(p + 1) * lda + i]);
          MAT_NEON_TYPE a11 = MAT_NEON_LOAD(&A[(p + 1) * lda + i + NW]);
          acc00 = MAT_NEON_FMA_LANE(acc00, a10, b1, 0);
          acc01 = MAT_NEON_FMA_LANE(acc01, a11, b1, 0);
          acc10 = MAT_NEON_FMA_LANE(acc10, a10, b1, 1);
          acc11 = MAT_NEON_FMA_LANE(acc11, a11, b1, 1);
          acc20 = MAT_NEON_FMA_LANE(acc20, a10, b1, 2);
          acc21 = MAT_NEON_FMA_LANE(acc21, a11, b1, 2);
          acc30 = MAT_NEON_FMA_LANE(acc30, a10, b1, 3);
          acc31 = MAT_NEON_FMA_LANE(acc31, a11, b1, 3);

          MAT_NEON_TYPE a20 = MAT_NEON_LOAD(&A[(p + 2) * lda + i]);
          MAT_NEON_TYPE a21 = MAT_NEON_LOAD(&A[(p + 2) * lda + i + NW]);
          acc00 = MAT_NEON_FMA_LANE(acc00, a20, b2, 0);
          acc01 = MAT_NEON_FMA_LANE(acc01, a21, b2, 0);
          acc10 = MAT_NEON_FMA_LANE(acc10, a20, b2, 1);
          acc11 = MAT_NEON_FMA_LANE(acc11, a21, b2, 1);
          acc20 = MAT_NEON_FMA_LANE(acc20, a20, b2, 2);
          acc21 = MAT_NEON_FMA_LANE(acc21, a21, b2, 2);
          acc30 = MAT_NEON_FMA_LANE(acc30, a20, b2, 3);
          acc31 = MAT_NEON_FMA_LANE(acc31, a21, b2, 3);

          MAT_NEON_TYPE a30 = MAT_NEON_LOAD(&A[(p + 3) * lda + i]);
          MAT_NEON_TYPE a31 = MAT_NEON_LOAD(&A[(p + 3) * lda + i + NW]);
          acc00 = MAT_NEON_FMA_LANE(acc00, a30, b3, 0);
          acc01 = MAT_NEON_FMA_LANE(acc01, a31, b3, 0);
          acc10 = MAT_NEON_FMA_LANE(acc10, a30, b3, 1);
          acc11 = MAT_NEON_FMA_LANE(acc11, a31, b3, 1);
          acc20 = MAT_NEON_FMA_LANE(acc20, a30, b3, 2);
          acc21 = MAT_NEON_FMA_LANE(acc21, a31, b3, 2);
          acc30 = MAT_NEON_FMA_LANE(acc30, a30, b3, 3);
          acc31 = MAT_NEON_FMA_LANE(acc31, a31, b3, 3);
        }
        for (; p < k_end; p++) {
          MAT_NEON_TYPE a0 = MAT_NEON_LOAD(&A[p * lda + i]);
          MAT_NEON_TYPE a1 = MAT_NEON_LOAD(&A[p * lda + i + NW]);
          MAT_NEON_TYPE b = MAT_NEON_LOAD(&A[p * lda + j]);
          acc00 = MAT_NEON_FMA_LANE(acc00, a0, b, 0);
          acc01 = MAT_NEON_FMA_LANE(acc01, a1, b, 0);
          acc10 = MAT_NEON_FMA_LANE(acc10, a0, b, 1);
          acc11 = MAT_NEON_FMA_LANE(acc11, a1, b, 1);
          acc20 = MAT_NEON_FMA_LANE(acc20, a0, b, 2);
          acc21 = MAT_NEON_FMA_LANE(acc21, a1, b, 2);
          acc30 = MAT_NEON_FMA_LANE(acc30, a0, b, 3);
          acc31 = MAT_NEON_FMA_LANE(acc31, a1, b, 3);
        }

        // Accumulate into C (beta already applied)
        mat_elem_t *C0 = &C[(j + 0) * ldc + i];
        mat_elem_t *C1 = &C[(j + 1) * ldc + i];
        mat_elem_t *C2 = &C[(j + 2) * ldc + i];
        mat_elem_t *C3 = &C[(j + 3) * ldc + i];
        MAT_NEON_STORE(C0, MAT_NEON_FMA(MAT_NEON_LOAD(C0), valpha, acc00));
        MAT_NEON_STORE(C0 + NW,
                       MAT_NEON_FMA(MAT_NEON_LOAD(C0 + NW), valpha, acc01));
        MAT_NEON_STORE(C1, MAT_NEON_FMA(MAT_NEON_LOAD(C1), valpha, acc10));
        MAT_NEON_STORE(C1 + NW,
                       MAT_NEON_FMA(MAT_NEON_LOAD(C1 + NW), valpha, acc11));
        MAT_NEON_STORE(C2, MAT_NEON_FMA(MAT_NEON_LOAD(C2), valpha, acc20));
        MAT_NEON_STORE(C2 + NW,
                       MAT_NEON_FMA(MAT_NEON_LOAD(C2 + NW), valpha, acc21));
        MAT_NEON_STORE(C3, MAT_NEON_FMA(MAT_NEON_LOAD(C3), valpha, acc30));
        MAT_NEON_STORE(C3 + NW,
                       MAT_NEON_FMA(MAT_NEON_LOAD(C3 + NW), valpha, acc31));
      }

      // 4x4 blocks
      for (; i + NW <= m; i += NW) {
        MAT_NEON_TYPE acc0 = MAT_NEON_DUP(0), acc1 = MAT_NEON_DUP(0);
        MAT_NEON_TYPE acc2 = MAT_NEON_DUP(0), acc3 = MAT_NEON_DUP(0);

        size_t p = kb;
        for (; p + 4 <= k_end; p += 4) {
          MAT_NEON_TYPE b0 = MAT_NEON_LOAD(&A[(p + 0) * lda + j]);
          MAT_NEON_TYPE b1 = MAT_NEON_LOAD(&A[(p + 1) * lda + j]);
          MAT_NEON_TYPE b2 = MAT_NEON_LOAD(&A[(p + 2) * lda + j]);
          MAT_NEON_TYPE b3 = MAT_NEON_LOAD(&A[(p + 3) * lda + j]);
          MAT_NEON_TYPE a0 = MAT_NEON_LOAD(&A[(p + 0) * lda + i]);
          MAT_NEON_TYPE a1 = MAT_NEON_LOAD(&A[(p + 1) * lda + i]);
          MAT_NEON_TYPE a2 = MAT_NEON_LOAD(&A[(p + 2) * lda + i]);
          MAT_NEON_TYPE a3 = MAT_NEON_LOAD(&A[(p + 3) * lda + i]);
          acc0 = MAT_NEON_FMA_LANE(acc0, a0, b0, 0);
          acc0 = MAT_NEON_FMA_LANE(acc0, a1, b1, 0);
          acc0 = MAT_NEON_FMA_LANE(acc0, a2, b2, 0);
          acc0 = MAT_NEON_FMA_LANE(acc0, a3, b3, 0);
          acc1 = MAT_NEON_FMA_LANE(acc1, a0, b0, 1);
          acc1 = MAT_NEON_FMA_LANE(acc1, a1, b1, 1);
          acc1 = MAT_NEON_FMA_LANE(acc1, a2, b2, 1);
          acc1 = MAT_NEON_FMA_LANE(acc1, a3, b3, 1);
          acc2 = MAT_NEON_FMA_LANE(acc2, a0, b0, 2);
          acc2 = MAT_NEON_FMA_LANE(acc2, a1, b1, 2);
          acc2 = MAT_NEON_FMA_LANE(acc2, a2, b2, 2);
          acc2 = MAT_NEON_FMA_LANE(acc2, a3, b3, 2);
          acc3 = MAT_NEON_FMA_LANE(acc3, a0, b0, 3);
          acc3 = MAT_NEON_FMA_LANE(acc3, a1, b1, 3);
          acc3 = MAT_NEON_FMA_LANE(acc3, a2, b2, 3);
          acc3 = MAT_NEON_FMA_LANE(acc3, a3, b3, 3);
        }
        for (; p < k_end; p++) {
          MAT_NEON_TYPE a = MAT_NEON_LOAD(&A[p * lda + i]);
          MAT_NEON_TYPE b = MAT_NEON_LOAD(&A[p * lda + j]);
          acc0 = MAT_NEON_FMA_LANE(acc0, a, b, 0);
          acc1 = MAT_NEON_FMA_LANE(acc1, a, b, 1);
          acc2 = MAT_NEON_FMA_LANE(acc2, a, b, 2);
          acc3 = MAT_NEON_FMA_LANE(acc3, a, b, 3);
        }

        mat_elem_t *C0 = &C[(j + 0) * ldc + i];
        mat_elem_t *C1 = &C[(j + 1) * ldc + i];
        mat_elem_t *C2 = &C[(j + 2) * ldc + i];
        mat_elem_t *C3 = &C[(j + 3) * ldc + i];
        MAT_NEON_STORE(C0, MAT_NEON_FMA(MAT_NEON_LOAD(C0), valpha, acc0));
        MAT_NEON_STORE(C1, MAT_NEON_FMA(MAT_NEON_LOAD(C1), valpha, acc1));
        MAT_NEON_STORE(C2, MAT_NEON_FMA(MAT_NEON_LOAD(C2), valpha, acc2));
        MAT_NEON_STORE(C3, MAT_NEON_FMA(MAT_NEON_LOAD(C3), valpha, acc3));
      }

      // Remaining rows
      for (; i < m; i++) {
        for (size_t jj = j; jj < j + NW; jj++) {
          mat_elem_t sum = 0;
          for (size_t p = kb; p < k_end; p++)
            sum += A[p * lda + i] * A[p * lda + jj];
          C[jj * ldc + i] += alpha * sum;
        }
      }

      // Diagonal block
      {
        MAT_NEON_TYPE acc0 = MAT_NEON_DUP(0), acc1 = MAT_NEON_DUP(0);
        MAT_NEON_TYPE acc2 = MAT_NEON_DUP(0), acc3 = MAT_NEON_DUP(0);

        size_t p = kb;
        for (; p + 4 <= k_end; p += 4) {
          MAT_NEON_TYPE b0 = MAT_NEON_LOAD(&A[(p + 0) * lda + j]);
          MAT_NEON_TYPE b1 = MAT_NEON_LOAD(&A[(p + 1) * lda + j]);
          MAT_NEON_TYPE b2 = MAT_NEON_LOAD(&A[(p + 2) * lda + j]);
          MAT_NEON_TYPE b3 = MAT_NEON_LOAD(&A[(p + 3) * lda + j]);
          acc0 = MAT_NEON_FMA_LANE(acc0, b0, b0, 0);
          acc0 = MAT_NEON_FMA_LANE(acc0, b1, b1, 0);
          acc0 = MAT_NEON_FMA_LANE(acc0, b2, b2, 0);
          acc0 = MAT_NEON_FMA_LANE(acc0, b3, b3, 0);
          acc1 = MAT_NEON_FMA_LANE(acc1, b0, b0, 1);
          acc1 = MAT_NEON_FMA_LANE(acc1, b1, b1, 1);
          acc1 = MAT_NEON_FMA_LANE(acc1, b2, b2, 1);
          acc1 = MAT_NEON_FMA_LANE(acc1, b3, b3, 1);
          acc2 = MAT_NEON_FMA_LANE(acc2, b0, b0, 2);
          acc2 = MAT_NEON_FMA_LANE(acc2, b1, b1, 2);
          acc2 = MAT_NEON_FMA_LANE(acc2, b2, b2, 2);
          acc2 = MAT_NEON_FMA_LANE(acc2, b3, b3, 2);
          acc3 = MAT_NEON_FMA_LANE(acc3, b0, b0, 3);
          acc3 = MAT_NEON_FMA_LANE(acc3, b1, b1, 3);
          acc3 = MAT_NEON_FMA_LANE(acc3, b2, b2, 3);
          acc3 = MAT_NEON_FMA_LANE(acc3, b3, b3, 3);
        }
        for (; p < k_end; p++) {
          MAT_NEON_TYPE b = MAT_NEON_LOAD(&A[p * lda + j]);
          acc0 = MAT_NEON_FMA_LANE(acc0, b, b, 0);
          acc1 = MAT_NEON_FMA_LANE(acc1, b, b, 1);
          acc2 = MAT_NEON_FMA_LANE(acc2, b, b, 2);
          acc3 = MAT_NEON_FMA_LANE(acc3, b, b, 3);
        }

        mat_elem_t s[4][4];
        MAT_NEON_STORE(s[0], acc0);
        MAT_NEON_STORE(s[1], acc1);
        MAT_NEON_STORE(s[2], acc2);
        MAT_NEON_STORE(s[3], acc3);

        for (size_t col = 0; col < NW; col++) {
          for (size_t row = col; row < NW; row++) {
            C[(j + col) * ldc + (j + row)] += alpha * s[col][row];
          }
        }
      }
    }

    // Remaining columns
    for (; j < m; j++) {
      for (size_t i = j; i < m; i++) {
        mat_elem_t sum = 0;
        for (size_t p = kb; p < k_end; p++)
          sum += A[p * lda + i] * A[p * lda + j];
        C[j * ldc + i] += alpha * sum;
      }
    }
  }
}
#undef MAT_SYRK_K_BLOCK
#endif

// Column-major SYRK lower triangle dispatcher
// Use unblocked for small problems, blocked for larger ones
#define MAT_SYRK_BLOCK_THRESHOLD 192
MAT_INTERNAL_STATIC void mat_syrk_lower_colmajor_(
    mat_elem_t *C, size_t ldc, mat_elem_t alpha, const mat_elem_t *A,
    size_t lda, size_t m, size_t k, mat_elem_t beta) {
#ifdef MAT_HAS_ARM_NEON
  if (m >= MAT_SYRK_BLOCK_THRESHOLD) {
    mat_syrk_lower_colmajor_blocked_(C, ldc, alpha, A, lda, m, k, beta);
  } else {
    mat_syrk_lower_colmajor_unblocked_(C, ldc, alpha, A, lda, m, k, beta);
  }
#else
  mat_syrk_lower_colmajor_scalar_(C, ldc, alpha, A, lda, m, k, beta);
#endif
}
#undef MAT_SYRK_BLOCK_THRESHOLD

// Transpose: At[i,p] = A[p,i] - Scalar implementation
MAT_INTERNAL_STATIC void
mat_transpose_block_scalar_(mat_elem_t *At, size_t ldat, const mat_elem_t *A,
                            size_t lda, size_t rows, size_t cols) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t p = 0; p < cols; p++) {
      At[i * ldat + p] = A[p * lda + i];
    }
  }
}

#ifdef MAT_HAS_ARM_NEON
// Transpose: At[i,p] = A[p,i] - NEON implementation
// Uses MAT_NEON_WIDTH x MAT_NEON_WIDTH block transpose with ZIP operations
MAT_INTERNAL_STATIC void mat_transpose_block_neon_(mat_elem_t *At, size_t ldat,
                                                   const mat_elem_t *A,
                                                   size_t lda, size_t rows,
                                                   size_t cols) {
  size_t i = 0;
  for (; i + MAT_NEON_WIDTH <= rows; i += MAT_NEON_WIDTH) {
    size_t p = 0;
    for (; p + MAT_NEON_WIDTH <= cols; p += MAT_NEON_WIDTH) {
      // Load MAT_NEON_WIDTH rows, each with MAT_NEON_WIDTH elements
      MAT_NEON_TYPE r0 = MAT_NEON_LOAD(&A[(p + 0) * lda + i]);
      MAT_NEON_TYPE r1 = MAT_NEON_LOAD(&A[(p + 1) * lda + i]);
#if MAT_NEON_WIDTH == 2
      // 2x2 transpose for double precision
      MAT_NEON_TYPE t0 = MAT_NEON_ZIP1(r0, r1);
      MAT_NEON_TYPE t1 = MAT_NEON_ZIP2(r0, r1);
      MAT_NEON_STORE(&At[(i + 0) * ldat + p], t0);
      MAT_NEON_STORE(&At[(i + 1) * ldat + p], t1);
#else
      // 4x4 transpose for single precision
      MAT_NEON_TYPE r2 = MAT_NEON_LOAD(&A[(p + 2) * lda + i]);
      MAT_NEON_TYPE r3 = MAT_NEON_LOAD(&A[(p + 3) * lda + i]);
      // First stage: zip pairs
      MAT_NEON_TYPE z01_lo = MAT_NEON_ZIP1(r0, r1);
      MAT_NEON_TYPE z01_hi = MAT_NEON_ZIP2(r0, r1);
      MAT_NEON_TYPE z23_lo = MAT_NEON_ZIP1(r2, r3);
      MAT_NEON_TYPE z23_hi = MAT_NEON_ZIP2(r2, r3);
      // Second stage: zip again to complete transpose
      MAT_NEON_TYPE t0 = MAT_NEON_ZIP1(z01_lo, z23_lo);
      MAT_NEON_TYPE t1 = MAT_NEON_ZIP2(z01_lo, z23_lo);
      MAT_NEON_TYPE t2 = MAT_NEON_ZIP1(z01_hi, z23_hi);
      MAT_NEON_TYPE t3 = MAT_NEON_ZIP2(z01_hi, z23_hi);
      MAT_NEON_STORE(&At[(i + 0) * ldat + p], t0);
      MAT_NEON_STORE(&At[(i + 1) * ldat + p], t1);
      MAT_NEON_STORE(&At[(i + 2) * ldat + p], t2);
      MAT_NEON_STORE(&At[(i + 3) * ldat + p], t3);
#endif
    }
    // Remainder columns
    for (; p < cols; p++) {
      for (size_t di = 0; di < MAT_NEON_WIDTH; di++) {
        At[(i + di) * ldat + p] = A[p * lda + i + di];
      }
    }
  }
  // Remainder rows
  for (; i < rows; i++) {
    for (size_t p = 0; p < cols; p++) {
      At[i * ldat + p] = A[p * lda + i];
    }
  }
}
#endif

// Transpose: At[i,p] = A[p,i]
MAT_INTERNAL_STATIC void mat_transpose_block_(mat_elem_t *At, size_t ldat,
                                              const mat_elem_t *A, size_t lda,
                                              size_t rows, size_t cols) {
#ifdef MAT_HAS_ARM_NEON
  mat_transpose_block_neon_(At, ldat, A, lda, rows, cols);
#else
  mat_transpose_block_scalar_(At, ldat, A, lda, rows, cols);
#endif
}

// Symmetric rank-k update for lower triangle (transposed): C = alpha * A^T * A
// + beta * C A is k x n, C is n x n Strategy: fast NEON transpose then
// optimized SYRK
MAT_INTERNAL_STATIC void mat_syrk_t_lower_(mat_elem_t *C, size_t ldc,
                                           mat_elem_t alpha,
                                           const mat_elem_t *A, size_t lda,
                                           size_t n, size_t k,
                                           mat_elem_t beta) {
#ifndef MAT_NO_SCRATCH
  mat_elem_t *At = (mat_elem_t *)mat_scratch_alloc_(n * k * sizeof(mat_elem_t));
#else
  mat_elem_t *At = (mat_elem_t *)mat_scratch_alloc_(n * k * sizeof(mat_elem_t));
#endif

  // Fast blocked transpose
  mat_transpose_block_(At, k, A, lda, n, k);

  // Use optimized lower SYRK
  mat_syrk_lower_(C, ldc, alpha, At, k, n, k, beta);

#ifndef MAT_NO_SCRATCH
  mat_scratch_reset_();
#else
  mat_scratch_free_(At);
#endif
}

// Symmetric rank-k update for upper triangle (transposed): C = alpha * A^T * A
// + beta * C A is k x n, C is n x n
MAT_INTERNAL_STATIC void mat_syrk_t_upper_(mat_elem_t *C, size_t ldc,
                                           mat_elem_t alpha,
                                           const mat_elem_t *A, size_t lda,
                                           size_t n, size_t k,
                                           mat_elem_t beta) {
#ifndef MAT_NO_SCRATCH
  mat_elem_t *At = (mat_elem_t *)mat_scratch_alloc_(n * k * sizeof(mat_elem_t));
#else
  mat_elem_t *At = (mat_elem_t *)mat_scratch_alloc_(n * k * sizeof(mat_elem_t));
#endif

  // Fast blocked transpose
  mat_transpose_block_(At, k, A, lda, n, k);

  // Use optimized upper SYRK
  mat_syrk_upper_(C, ldc, alpha, At, k, n, k, beta);

#ifndef MAT_NO_SCRATCH
  mat_scratch_reset_();
#else
  mat_scratch_free_(At);
#endif
}

// Column-major SYRK: transpose A to row-major, use optimized kernel
// The transpose cost O(n*k) is cheaper than cache miss penalty of strided access
#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void mat_syrk_colmajor_neon_(Mat *C, const Mat *A,
                                                  mat_elem_t alpha,
                                                  mat_elem_t beta, char uplo) {
  size_t n = C->rows;
  size_t k = A->cols;

  // Transpose A from column-major to row-major for cache-friendly access
  // Column-major A[i,kk] = A->data[kk * n + i]
  // Row-major At[i,kk] = At[i * k + kk]
  mat_elem_t *At =
      (mat_elem_t *)mat_scratch_alloc_(n * k * sizeof(mat_elem_t));
  const mat_elem_t *pa = A->data;
  for (size_t i = 0; i < n; i++) {
    for (size_t kk = 0; kk < k; kk++) {
      At[i * k + kk] = pa[kk * n + i];
    }
  }

  // For symmetric matrices:
  // Row-major C[i,j] at position i*n+j = Column-major C[j,i] at same position
  // So row-major lower (i >= j) = column-major upper (j <= i)
  // Therefore: swap uplo when calling row-major kernel
  if (uplo == 'L' || uplo == 'l') {
    // Column-major lower -> use row-major upper
    mat_syrk_upper_neon_(C->data, n, alpha, At, k, n, k, beta);
  } else {
    // Column-major upper -> use row-major lower
    mat_syrk_lower_neon_(C->data, n, alpha, At, k, n, k, beta);
  }

#ifndef MAT_NO_SCRATCH
  mat_scratch_reset_();
#else
  mat_scratch_free_(At);
#endif
}
#endif

// Generic scalar SYRK using MAT_AT/MAT_SET (works with any storage order)
MAT_INTERNAL_STATIC void mat_syrk_generic_(Mat *C, const Mat *A,
                                           mat_elem_t alpha, mat_elem_t beta,
                                           char uplo) {
  size_t n = C->rows;
  size_t k = A->cols;

  if (uplo == 'L' || uplo == 'l') {
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j <= i; j++) {
        mat_elem_t sum = 0;
        for (size_t kk = 0; kk < k; kk++) {
          sum += MAT_AT(A, i, kk) * MAT_AT(A, j, kk);
        }
        MAT_SET(C, i, j, beta * MAT_AT(C, i, j) + alpha * sum);
      }
    }
  } else {
    for (size_t i = 0; i < n; i++) {
      for (size_t j = i; j < n; j++) {
        mat_elem_t sum = 0;
        for (size_t kk = 0; kk < k; kk++) {
          sum += MAT_AT(A, i, kk) * MAT_AT(A, j, kk);
        }
        MAT_SET(C, i, j, beta * MAT_AT(C, i, j) + alpha * sum);
      }
    }
  }
}

MATDEF void mat_syrk(Mat *C, const Mat *A, mat_elem_t alpha, mat_elem_t beta,
                     char uplo) {
  MAT_ASSERT_MAT(C);
  MAT_ASSERT_MAT(A);
  MAT_ASSERT(C->rows == C->cols);
  MAT_ASSERT(C->rows == A->rows);

#ifdef MAT_HAS_ARM_NEON
  mat_syrk_colmajor_neon_(C, A, alpha, beta, uplo);
#else
  mat_syrk_generic_(C, A, alpha, beta, uplo);
#endif
}

// Column-major SYRK_T: C = A^T * A
// In column-major, columns of A are contiguous, so we can use row-major kernel
// directly (interpreting columns as rows)
#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void mat_syrk_t_colmajor_neon_(Mat *C, const Mat *A,
                                                    mat_elem_t alpha,
                                                    mat_elem_t beta, char uplo) {
  size_t n = C->rows;  // = A->cols
  size_t k = A->rows;  // inner dimension

  // In column-major: column i of A is at A->data[i * k + 0..k-1] (contiguous)
  // This looks like row-major A with n rows of length k, lda = k
  // C[i,j] = dot(column_i, column_j) = dot(row_i, row_j) in this interpretation

  // Still need to swap uplo for the column-major C output
  if (uplo == 'L' || uplo == 'l') {
    // Column-major lower -> use row-major upper
    mat_syrk_upper_neon_(C->data, n, alpha, A->data, k, n, k, beta);
  } else {
    // Column-major upper -> use row-major lower
    mat_syrk_lower_neon_(C->data, n, alpha, A->data, k, n, k, beta);
  }
}
#endif

// Generic scalar SYRK_T using MAT_AT/MAT_SET (works with any storage order)
// Computes C = alpha * A^T * A + beta * C
MAT_INTERNAL_STATIC void mat_syrk_t_generic_(Mat *C, const Mat *A,
                                             mat_elem_t alpha, mat_elem_t beta,
                                             char uplo) {
  size_t n = C->rows;
  size_t k = A->rows;

  if (uplo == 'L' || uplo == 'l') {
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j <= i; j++) {
        mat_elem_t sum = 0;
        for (size_t kk = 0; kk < k; kk++) {
          sum += MAT_AT(A, kk, i) * MAT_AT(A, kk, j);
        }
        MAT_SET(C, i, j, beta * MAT_AT(C, i, j) + alpha * sum);
      }
    }
  } else {
    for (size_t i = 0; i < n; i++) {
      for (size_t j = i; j < n; j++) {
        mat_elem_t sum = 0;
        for (size_t kk = 0; kk < k; kk++) {
          sum += MAT_AT(A, kk, i) * MAT_AT(A, kk, j);
        }
        MAT_SET(C, i, j, beta * MAT_AT(C, i, j) + alpha * sum);
      }
    }
  }
}

MATDEF void mat_syrk_t(Mat *C, const Mat *A, mat_elem_t alpha, mat_elem_t beta,
                       char uplo) {
  MAT_ASSERT_MAT(C);
  MAT_ASSERT_MAT(A);
  MAT_ASSERT(C->rows == C->cols);
  MAT_ASSERT(C->rows == A->cols);

#ifdef MAT_HAS_ARM_NEON
  mat_syrk_t_colmajor_neon_(C, A, alpha, beta, uplo);
#else
  mat_syrk_t_generic_(C, A, alpha, beta, uplo);
#endif
}

// Column-major unblocked Cholesky using dot products for better cache locality
// In column-major: L[0:j, i] is contiguous at M[i*ldm], so dots are efficient
MAT_INTERNAL_STATIC int mat_chol_unblocked_colmajor_(mat_elem_t *M, size_t n,
                                                      size_t ldm) {
  for (size_t j = 0; j < n; j++) {
    // Diagonal: L[j,j] = sqrt(A[j,j] - dot(L[0:j, j], L[0:j, j]))
    // L[0:j, j] is at M[j*ldm + 0:j] which is contiguous
    mat_elem_t *Lj = &M[j * ldm];  // Column j, elements 0:n
    mat_elem_t diag_sum = mat_dot_raw_(Lj, Lj, j);
    mat_elem_t diag = Lj[j] - diag_sum;

    if (diag <= 0) {
      return -1;
    }
    mat_elem_t ljj = MAT_SQRT(diag);
    Lj[j] = ljj;
    mat_elem_t ljj_inv = 1.0f / ljj;

    // Off-diagonal: L[i,j] = (A[i,j] - dot(L[0:j, i], L[0:j, j])) / L[j,j]
    // L[0:j, i] is at M[i*ldm + 0:j] which is contiguous
#ifdef MAT_HAS_ARM_NEON
    // Process 4 rows at a time for better throughput
    size_t i = j + 1;
    for (; i + 4 <= n; i += 4) {
      mat_elem_t *Li0 = &M[(i + 0) * ldm];
      mat_elem_t *Li1 = &M[(i + 1) * ldm];
      mat_elem_t *Li2 = &M[(i + 2) * ldm];
      mat_elem_t *Li3 = &M[(i + 3) * ldm];

      // Compute 4 dot products in parallel
      MAT_NEON_TYPE sum0 = MAT_NEON_DUP(0), sum1 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE sum2 = MAT_NEON_DUP(0), sum3 = MAT_NEON_DUP(0);

      size_t k = 0;
      for (; k + MAT_NEON_WIDTH <= j; k += MAT_NEON_WIDTH) {
        MAT_NEON_TYPE vLj = MAT_NEON_LOAD(&Lj[k]);
        sum0 = MAT_NEON_FMA(sum0, MAT_NEON_LOAD(&Li0[k]), vLj);
        sum1 = MAT_NEON_FMA(sum1, MAT_NEON_LOAD(&Li1[k]), vLj);
        sum2 = MAT_NEON_FMA(sum2, MAT_NEON_LOAD(&Li2[k]), vLj);
        sum3 = MAT_NEON_FMA(sum3, MAT_NEON_LOAD(&Li3[k]), vLj);
      }

      mat_elem_t d0 = MAT_NEON_ADDV(sum0);
      mat_elem_t d1 = MAT_NEON_ADDV(sum1);
      mat_elem_t d2 = MAT_NEON_ADDV(sum2);
      mat_elem_t d3 = MAT_NEON_ADDV(sum3);

      // Scalar tail for dot products
      for (; k < j; k++) {
        mat_elem_t lk = Lj[k];
        d0 += Li0[k] * lk;
        d1 += Li1[k] * lk;
        d2 += Li2[k] * lk;
        d3 += Li3[k] * lk;
      }

      // Store results in column j
      Lj[i + 0] = (Lj[i + 0] - d0) * ljj_inv;
      Lj[i + 1] = (Lj[i + 1] - d1) * ljj_inv;
      Lj[i + 2] = (Lj[i + 2] - d2) * ljj_inv;
      Lj[i + 3] = (Lj[i + 3] - d3) * ljj_inv;
    }
    // Remaining rows
    for (; i < n; i++) {
      mat_elem_t *Li = &M[i * ldm];
      mat_elem_t dot = mat_dot_raw_(Li, Lj, j);
      Lj[i] = (Lj[i] - dot) * ljj_inv;
    }
#else
    for (size_t i = j + 1; i < n; i++) {
      mat_elem_t *Li = &M[i * ldm];
      mat_elem_t dot = mat_dot_raw_(Li, Lj, j);
      Lj[i] = (Lj[i] - dot) * ljj_inv;
    }
#endif
  }
  return 0;
}

// Legacy AXPY-based implementation for reference
MAT_INTERNAL_STATIC int mat_chol_unblocked_colmajor_axpy_(mat_elem_t *M, size_t n,
                                                          size_t ldm) {
  for (size_t j = 0; j < n; j++) {
    mat_elem_t *colj = &M[j * ldm];

    for (size_t k = 0; k < j; k++) {
      mat_elem_t ljk = M[k * ldm + j];
      mat_elem_t *colk = &M[k * ldm];

#ifdef MAT_HAS_ARM_NEON
      MAT_NEON_TYPE vljk = MAT_NEON_DUP(ljk);
      size_t i = j;
      for (; i + MAT_NEON_WIDTH * 4 <= n; i += MAT_NEON_WIDTH * 4) {
        MAT_NEON_TYPE c0 = MAT_NEON_LOAD(&colj[i]);
        MAT_NEON_TYPE c1 = MAT_NEON_LOAD(&colj[i + MAT_NEON_WIDTH]);
        MAT_NEON_TYPE c2 = MAT_NEON_LOAD(&colj[i + MAT_NEON_WIDTH * 2]);
        MAT_NEON_TYPE c3 = MAT_NEON_LOAD(&colj[i + MAT_NEON_WIDTH * 3]);
        MAT_NEON_TYPE k0 = MAT_NEON_LOAD(&colk[i]);
        MAT_NEON_TYPE k1 = MAT_NEON_LOAD(&colk[i + MAT_NEON_WIDTH]);
        MAT_NEON_TYPE k2 = MAT_NEON_LOAD(&colk[i + MAT_NEON_WIDTH * 2]);
        MAT_NEON_TYPE k3 = MAT_NEON_LOAD(&colk[i + MAT_NEON_WIDTH * 3]);
        c0 = MAT_NEON_SUB(c0, MAT_NEON_MUL(vljk, k0));
        c1 = MAT_NEON_SUB(c1, MAT_NEON_MUL(vljk, k1));
        c2 = MAT_NEON_SUB(c2, MAT_NEON_MUL(vljk, k2));
        c3 = MAT_NEON_SUB(c3, MAT_NEON_MUL(vljk, k3));
        MAT_NEON_STORE(&colj[i], c0);
        MAT_NEON_STORE(&colj[i + MAT_NEON_WIDTH], c1);
        MAT_NEON_STORE(&colj[i + MAT_NEON_WIDTH * 2], c2);
        MAT_NEON_STORE(&colj[i + MAT_NEON_WIDTH * 3], c3);
      }
      for (; i + MAT_NEON_WIDTH <= n; i += MAT_NEON_WIDTH) {
        MAT_NEON_TYPE c = MAT_NEON_LOAD(&colj[i]);
        MAT_NEON_TYPE k = MAT_NEON_LOAD(&colk[i]);
        MAT_NEON_STORE(&colj[i], MAT_NEON_SUB(c, MAT_NEON_MUL(vljk, k)));
      }
      for (; i < n; i++) {
        colj[i] -= ljk * colk[i];
      }
#else
      for (size_t i = j; i < n; i++) {
        colj[i] -= ljk * colk[i];
      }
#endif
    }

    if (colj[j] <= 0) {
      return -1;
    }
    mat_elem_t ljj = MAT_SQRT(colj[j]);
    colj[j] = ljj;

    mat_elem_t ljj_inv = 1.0f / ljj;
#ifdef MAT_HAS_ARM_NEON
    MAT_NEON_TYPE vinv = MAT_NEON_DUP(ljj_inv);
    size_t i = j + 1;
    for (; i + MAT_NEON_WIDTH * 4 <= n; i += MAT_NEON_WIDTH * 4) {
      MAT_NEON_TYPE c0 = MAT_NEON_LOAD(&colj[i]);
      MAT_NEON_TYPE c1 = MAT_NEON_LOAD(&colj[i + MAT_NEON_WIDTH]);
      MAT_NEON_TYPE c2 = MAT_NEON_LOAD(&colj[i + MAT_NEON_WIDTH * 2]);
      MAT_NEON_TYPE c3 = MAT_NEON_LOAD(&colj[i + MAT_NEON_WIDTH * 3]);
      MAT_NEON_STORE(&colj[i], MAT_NEON_MUL(c0, vinv));
      MAT_NEON_STORE(&colj[i + MAT_NEON_WIDTH], MAT_NEON_MUL(c1, vinv));
      MAT_NEON_STORE(&colj[i + MAT_NEON_WIDTH * 2], MAT_NEON_MUL(c2, vinv));
      MAT_NEON_STORE(&colj[i + MAT_NEON_WIDTH * 3], MAT_NEON_MUL(c3, vinv));
    }
    for (; i + MAT_NEON_WIDTH <= n; i += MAT_NEON_WIDTH) {
      MAT_NEON_TYPE c = MAT_NEON_LOAD(&colj[i]);
      MAT_NEON_STORE(&colj[i], MAT_NEON_MUL(c, vinv));
    }
    for (; i < n; i++) {
      colj[i] *= ljj_inv;
    }
#else
    for (size_t i = j + 1; i < n; i++) {
      colj[i] *= ljj_inv;
    }
#endif
  }
  return 0;
}

// Column-major blocked Cholesky (right-looking, uses mat_syrk for trailing update)
// Structure matches row-major: panel factorization -> TRSM -> SYRK
#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC int mat_chol_blocked_colmajor_(mat_elem_t *M, size_t n,
                                                    size_t ldm) {
  for (size_t kb = 0; kb < n; kb += MAT_CHOL_BLOCK_SIZE) {
    size_t k_end =
        (kb + MAT_CHOL_BLOCK_SIZE < n) ? kb + MAT_CHOL_BLOCK_SIZE : n;
    size_t block_k = k_end - kb;

    // 1. Panel factorization: unblocked Cholesky on diagonal block L11
    // Column-major: L11 starts at M[kb * ldm + kb]
    int result = mat_chol_unblocked_colmajor_axpy_(&M[kb * ldm + kb], block_k, ldm);
    if (result != 0) {
      return -1;
    }

    // 2. TRSM: Solve L21 from L11 * L21^T = A21^T
    // In column-major: L21[i,j] = M[(kb+j) * ldm + (k_end+i)]
    // Solve column-by-column: L21[:,j] = solve_trilt(L11, A21[:,j])
    if (k_end < n) {
      size_t trail_m = n - k_end;

      // For each column j of L21 (which is column kb+j of the full matrix)
      for (size_t j = 0; j < block_k; j++) {
        mat_elem_t *L11_col_j = &M[(kb + j) * ldm + kb];  // Column j of L11
        mat_elem_t *L21_col_j = &M[(kb + j) * ldm + k_end];  // Column j of L21

        // Forward substitution: solve L11 * x = A21[:,j]^T for x = L21[:,j]
        // But we need L11 * L21^T[j,:] = A21^T[j,:], which is:
        // For each row i of L21 (i.e., each element L21[i,j]):
        // L21[i,j] = (A21[i,j] - sum_{kk<j} L11[j,kk] * L21[i,kk]) / L11[j,j]
        mat_elem_t ljj_inv = 1.0f / L11_col_j[j];

        // Process all rows of L21 for this column j
        // First update using previous columns of L21
        for (size_t kk = 0; kk < j; kk++) {
          mat_elem_t L11_jk = M[(kb + kk) * ldm + (kb + j)];  // L11[j,kk]
          mat_elem_t *L21_col_kk = &M[(kb + kk) * ldm + k_end];

          // AXPY: L21[:,j] -= L11[j,kk] * L21[:,kk]
          MAT_NEON_TYPE vL11 = MAT_NEON_DUP(L11_jk);
          size_t i = 0;
          for (; i + MAT_NEON_WIDTH * 4 <= trail_m; i += MAT_NEON_WIDTH * 4) {
            MAT_NEON_TYPE c0 = MAT_NEON_LOAD(&L21_col_j[i]);
            MAT_NEON_TYPE c1 = MAT_NEON_LOAD(&L21_col_j[i + MAT_NEON_WIDTH]);
            MAT_NEON_TYPE c2 = MAT_NEON_LOAD(&L21_col_j[i + MAT_NEON_WIDTH * 2]);
            MAT_NEON_TYPE c3 = MAT_NEON_LOAD(&L21_col_j[i + MAT_NEON_WIDTH * 3]);
            c0 = MAT_NEON_SUB(c0, MAT_NEON_MUL(vL11, MAT_NEON_LOAD(&L21_col_kk[i])));
            c1 = MAT_NEON_SUB(c1, MAT_NEON_MUL(vL11, MAT_NEON_LOAD(&L21_col_kk[i + MAT_NEON_WIDTH])));
            c2 = MAT_NEON_SUB(c2, MAT_NEON_MUL(vL11, MAT_NEON_LOAD(&L21_col_kk[i + MAT_NEON_WIDTH * 2])));
            c3 = MAT_NEON_SUB(c3, MAT_NEON_MUL(vL11, MAT_NEON_LOAD(&L21_col_kk[i + MAT_NEON_WIDTH * 3])));
            MAT_NEON_STORE(&L21_col_j[i], c0);
            MAT_NEON_STORE(&L21_col_j[i + MAT_NEON_WIDTH], c1);
            MAT_NEON_STORE(&L21_col_j[i + MAT_NEON_WIDTH * 2], c2);
            MAT_NEON_STORE(&L21_col_j[i + MAT_NEON_WIDTH * 3], c3);
          }
          for (; i < trail_m; i++) {
            L21_col_j[i] -= L11_jk * L21_col_kk[i];
          }
        }

        // Scale by 1/L11[j,j]
        MAT_NEON_TYPE vinv = MAT_NEON_DUP(ljj_inv);
        size_t i = 0;
        for (; i + MAT_NEON_WIDTH * 4 <= trail_m; i += MAT_NEON_WIDTH * 4) {
          MAT_NEON_TYPE c0 = MAT_NEON_LOAD(&L21_col_j[i]);
          MAT_NEON_TYPE c1 = MAT_NEON_LOAD(&L21_col_j[i + MAT_NEON_WIDTH]);
          MAT_NEON_TYPE c2 = MAT_NEON_LOAD(&L21_col_j[i + MAT_NEON_WIDTH * 2]);
          MAT_NEON_TYPE c3 = MAT_NEON_LOAD(&L21_col_j[i + MAT_NEON_WIDTH * 3]);
          MAT_NEON_STORE(&L21_col_j[i], MAT_NEON_MUL(c0, vinv));
          MAT_NEON_STORE(&L21_col_j[i + MAT_NEON_WIDTH], MAT_NEON_MUL(c1, vinv));
          MAT_NEON_STORE(&L21_col_j[i + MAT_NEON_WIDTH * 2], MAT_NEON_MUL(c2, vinv));
          MAT_NEON_STORE(&L21_col_j[i + MAT_NEON_WIDTH * 3], MAT_NEON_MUL(c3, vinv));
        }
        for (; i < trail_m; i++) {
          L21_col_j[i] *= ljj_inv;
        }
      }

      // 3. SYRK: A22 -= L21 * L21^T (symmetric rank-k update on lower triangle)
      // Use native column-major SYRK - no transpose needed!
      // L21 is at M[kb*ldm + k_end] with stride ldm, dimensions trail_m × block_k
      // C22 is at M[k_end*ldm + k_end] with stride ldm, dimensions trail_m × trail_m
      mat_syrk_lower_colmajor_(&M[k_end * ldm + k_end], ldm, -1,
                               &M[kb * ldm + k_end], ldm,
                               trail_m, block_k, 1);
    }
  }
  return 0;
}
#endif

// Scalar Cholesky factorization using MAT_AT/MAT_SET (works with any storage)
MAT_INTERNAL_STATIC int mat_chol_scalar_generic_(Mat *L) {
  size_t n = L->rows;

  for (size_t j = 0; j < n; j++) {
    // Diagonal: L[j,j] = sqrt(A[j,j] - sum_{k<j} L[j,k]^2)
    mat_elem_t sum = 0;
    for (size_t k = 0; k < j; k++) {
      mat_elem_t ljk = MAT_AT(L, j, k);
      sum += ljk * ljk;
    }
    mat_elem_t diag = MAT_AT(L, j, j) - sum;

    if (diag <= 0) {
      return -1; // Not positive definite
    }
    mat_elem_t ljj = MAT_SQRT(diag);
    MAT_SET(L, j, j, ljj);
    mat_elem_t ljj_inv = 1.0f / ljj;

    // Off-diagonal: L[i,j] = (A[i,j] - dot(L[i,0:j], L[j,0:j])) / L[j,j]
    for (size_t i = j + 1; i < n; i++) {
      mat_elem_t dot = 0;
      for (size_t k = 0; k < j; k++) {
        dot += MAT_AT(L, i, k) * MAT_AT(L, j, k);
      }
      MAT_SET(L, i, j, (MAT_AT(L, i, j) - dot) * ljj_inv);
    }
  }
  return 0;
}

MATDEF int mat_chol(const Mat *A, Mat *L) {
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(L);
  MAT_ASSERT_SQUARE(A);

  size_t n = A->rows;
  MAT_ASSERT(L->rows == n && L->cols == n);

  // Column-major: iterate by column for contiguous memory access
  // Copy lower triangle of A to L, zero upper triangle
  for (size_t j = 0; j < n; j++) {
    mat_elem_t *Lcol = &L->data[j * n];
    const mat_elem_t *Acol = &A->data[j * n];
    // Zero upper triangle (rows 0 to j-1)
    for (size_t i = 0; i < j; i++) {
      Lcol[i] = 0;
    }
    // Copy lower triangle including diagonal (rows j to n-1)
    // Use memcpy for contiguous block
    memcpy(&Lcol[j], &Acol[j], (n - j) * sizeof(mat_elem_t));
  }

  // Column-major: use AXPY-based implementation (correct left-looking algorithm)
  mat_elem_t *Ldata = L->data;
#ifdef MAT_HAS_ARM_NEON
  if (n < MAT_CHOL_BLOCK_SIZE) {
    return mat_chol_unblocked_colmajor_axpy_(Ldata, n, n);
  } else {
    return mat_chol_blocked_colmajor_(Ldata, n, n);
  }
#else
  return mat_chol_scalar_generic_(L);
#endif
}

MATDEF void mat_inv(Mat *out, const Mat *A) {
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_SQUARE(A);
  size_t n = A->rows;
  MAT_ASSERT(out->rows == n && out->cols == n);

  Mat *L = mat_mat(n, n);
  Mat *U = mat_mat(n, n);
  Perm *p = mat_perm(n);

  mat_plu(A, L, U, p);

  // Column-major: use MAT_AT/MAT_SET for correct indexing
  // Initialize out = P * I (permuted identity)
  memset(out->data, 0, n * n * sizeof(mat_elem_t));
  for (size_t i = 0; i < n; i++)
    MAT_SET(out, i, p->data[i], 1);

  // Forward substitution: L * Y = P*I, solve for Y column by column
  for (size_t col = 0; col < n; col++) {
    for (size_t i = 0; i < n; i++) {
      mat_elem_t sum = MAT_AT(out, i, col);
      for (size_t j = 0; j < i; j++) {
        sum -= MAT_AT(L, i, j) * MAT_AT(out, j, col);
      }
      MAT_SET(out, i, col, sum);  // L[i,i] = 1 for unit lower triangular
    }
  }

  // Backward substitution: U * X = Y, solve for X column by column
  for (size_t col = 0; col < n; col++) {
    for (size_t i = n; i-- > 0;) {
      mat_elem_t sum = MAT_AT(out, i, col);
      for (size_t j = i + 1; j < n; j++) {
        sum -= MAT_AT(U, i, j) * MAT_AT(out, j, col);
      }
      MAT_SET(out, i, col, sum / MAT_AT(U, i, i));
    }
  }

  MAT_FREE_MAT(L);
  MAT_FREE_MAT(U);
  MAT_FREE_PERM(p);
}

MATDEF mat_elem_t mat_det(const Mat *A) {
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_SQUARE(A);
  size_t n = A->rows;

  Mat *L = mat_mat(n, n);
  Mat *U = mat_mat(n, n);
  Perm *p = mat_perm(n);

  int swaps = mat_plu(A, L, U, p);

  // det = (-1)^swaps * product of U diagonal
  mat_elem_t det = (swaps % 2 == 0) ? 1 : -1;
  for (size_t i = 0; i < n; i++) {
    det *= U->data[i * n + i];
  }

  MAT_FREE_MAT(L);
  MAT_FREE_MAT(U);
  MAT_FREE_PERM(p);

  return det;
}

// ============================================================================
// SVD: One-sided Jacobi algorithm
// ============================================================================

// Column-major SVD helpers - columns are contiguous for efficient SIMD
// W->data is column-major: W->data[row + col*m] = W[row, col]

#ifdef MAT_HAS_ARM_NEON
// NEON: Apply Jacobi rotation to columns i and j (column-major data)
MAT_INTERNAL_STATIC void mat_svd_rotate_cols_neon_(mat_elem_t *data, size_t m,
                                                   size_t i, size_t j,
                                                   mat_elem_t cs,
                                                   mat_elem_t sn) {

  mat_elem_t *col_i = &data[i * m];
  mat_elem_t *col_j = &data[j * m];

  MAT_NEON_TYPE vcs = MAT_NEON_DUP(cs);
  MAT_NEON_TYPE vsn = MAT_NEON_DUP(sn);

  size_t k = 0;
  for (; k + MAT_NEON_WIDTH <= m; k += MAT_NEON_WIDTH) {
    MAT_NEON_TYPE wi = MAT_NEON_LOAD(&col_i[k]);
    MAT_NEON_TYPE wj = MAT_NEON_LOAD(&col_j[k]);

    // new_i = cs * wi - sn * wj
    // new_j = sn * wi + cs * wj
    MAT_NEON_TYPE new_i = MAT_NEON_FMS(MAT_NEON_DUP(0), vsn, wj);
    new_i = MAT_NEON_FMA(new_i, vcs, wi);
    MAT_NEON_TYPE new_j = MAT_NEON_FMA(MAT_NEON_DUP(0), vsn, wi);
    new_j = MAT_NEON_FMA(new_j, vcs, wj);

    MAT_NEON_STORE(&col_i[k], new_i);
    MAT_NEON_STORE(&col_j[k], new_j);
  }

  for (; k < m; k++) {
    mat_elem_t wi = col_i[k];
    mat_elem_t wj = col_j[k];
    col_i[k] = cs * wi - sn * wj;
    col_j[k] = sn * wi + cs * wj;
  }
}

#endif // MAT_HAS_ARM_NEON

// Scalar: Apply Jacobi rotation (column-major data)
MAT_INTERNAL_STATIC void mat_svd_rotate_cols_scalar_(mat_elem_t *data, size_t m,
                                                     size_t i, size_t j,
                                                     mat_elem_t cs,
                                                     mat_elem_t sn) {
  mat_elem_t *col_i = &data[i * m];
  mat_elem_t *col_j = &data[j * m];
  for (size_t k = 0; k < m; k++) {
    mat_elem_t wi = col_i[k];
    mat_elem_t wj = col_j[k];
    col_i[k] = cs * wi - sn * wj;
    col_j[k] = sn * wi + cs * wj;
  }
}

// Compute Jacobi rotation parameters to zero out off-diagonal element.
// Given symmetric 2x2 matrix [[a, c], [c, b]], compute cos and sin such that
// the rotation diagonalizes it.
static void mat_svd_jacobi_rotation(mat_elem_t a, mat_elem_t b, mat_elem_t c,
                                    mat_elem_t *cs, mat_elem_t *sn) {
  if (MAT_FABS(c) < MAT_DEFAULT_EPSILON) {
    *cs = 1;
    *sn = 0;
    return;
  }

  mat_elem_t tau = (b - a) / (2 * c);
  mat_elem_t t;

  // Choose the smaller root for numerical stability
  if (tau >= 0) {
    t = 1 / (tau + MAT_SQRT(1 + tau * tau));
  } else {
    t = 1 / (tau - MAT_SQRT(1 + tau * tau));
  }

  *cs = 1 / MAT_SQRT(1 + t * t);
  *sn = t * (*cs);
}

// y += alpha * X[:,col] where y is contiguous Vec and X is row-major Mat
// Used for Gram-Schmidt on row-major matrices
#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void mat_svd_axpy_strided_neon_(Vec *y, mat_elem_t alpha,
                                                    const Mat *X, size_t col) {
  const mat_elem_t *x = &X->data[col];
  size_t x_stride = X->cols;
  size_t n = X->rows;
  mat_elem_t *yp = y->data;

  MAT_NEON_TYPE valpha = MAT_NEON_DUP(alpha);
  size_t i = 0;

  // Process 4 elements at a time using gather for x
  for (; i + MAT_NEON_WIDTH <= n; i += MAT_NEON_WIDTH) {
    MAT_NEON_TYPE vy = MAT_NEON_LOAD(&yp[i]);
    // Gather strided x elements into vector
#ifdef MAT_DOUBLE_PRECISION
    MAT_NEON_TYPE vx = {x[i * x_stride], x[(i + 1) * x_stride]};
#else
    MAT_NEON_TYPE vx = {x[i * x_stride], x[(i + 1) * x_stride],
                        x[(i + 2) * x_stride], x[(i + 3) * x_stride]};
#endif
    vy = MAT_NEON_FMA(vy, vx, valpha);
    MAT_NEON_STORE(&yp[i], vy);
  }

  // Scalar cleanup
  for (; i < n; i++) {
    yp[i] += alpha * x[i * x_stride];
  }
}
#endif

MAT_INTERNAL_STATIC void mat_svd_axpy_strided_scalar_(Vec *y, mat_elem_t alpha,
                                                      const Mat *X,
                                                      size_t col) {
  const mat_elem_t *x = &X->data[col];
  size_t x_stride = X->cols;
  size_t n = X->rows;

  for (size_t i = 0; i < n; i++) {
    y->data[i] += alpha * x[i * x_stride];
  }
}

// dot(y, X[:,col]) where y is contiguous Vec and X is row-major Mat
#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC mat_elem_t mat_svd_dot_strided_neon_(const Vec *y,
                                                         const Mat *X,
                                                         size_t col) {
  const mat_elem_t *yp = y->data;
  const mat_elem_t *x = &X->data[col];
  size_t x_stride = X->cols;
  size_t n = X->rows;

  MAT_NEON_TYPE vsum0 = MAT_NEON_DUP(0);
  MAT_NEON_TYPE vsum1 = MAT_NEON_DUP(0);
  size_t i = 0;

  // Process 4 elements at a time using gather for x
  for (; i + MAT_NEON_WIDTH * 2 <= n; i += MAT_NEON_WIDTH * 2) {
    MAT_NEON_TYPE vy0 = MAT_NEON_LOAD(&yp[i]);
    MAT_NEON_TYPE vy1 = MAT_NEON_LOAD(&yp[i + MAT_NEON_WIDTH]);
#ifdef MAT_DOUBLE_PRECISION
    MAT_NEON_TYPE vx0 = {x[i * x_stride], x[(i + 1) * x_stride]};
    MAT_NEON_TYPE vx1 = {x[(i + 2) * x_stride], x[(i + 3) * x_stride]};
#else
    MAT_NEON_TYPE vx0 = {x[i * x_stride], x[(i + 1) * x_stride],
                         x[(i + 2) * x_stride], x[(i + 3) * x_stride]};
    MAT_NEON_TYPE vx1 = {x[(i + 4) * x_stride], x[(i + 5) * x_stride],
                         x[(i + 6) * x_stride], x[(i + 7) * x_stride]};
#endif
    vsum0 = MAT_NEON_FMA(vsum0, vy0, vx0);
    vsum1 = MAT_NEON_FMA(vsum1, vy1, vx1);
  }

  for (; i + MAT_NEON_WIDTH <= n; i += MAT_NEON_WIDTH) {
    MAT_NEON_TYPE vy = MAT_NEON_LOAD(&yp[i]);
#ifdef MAT_DOUBLE_PRECISION
    MAT_NEON_TYPE vx = {x[i * x_stride], x[(i + 1) * x_stride]};
#else
    MAT_NEON_TYPE vx = {x[i * x_stride], x[(i + 1) * x_stride],
                        x[(i + 2) * x_stride], x[(i + 3) * x_stride]};
#endif
    vsum0 = MAT_NEON_FMA(vsum0, vy, vx);
  }

  vsum0 = MAT_NEON_ADD(vsum0, vsum1);
  mat_elem_t sum = MAT_NEON_ADDV(vsum0);

  // Scalar cleanup
  for (; i < n; i++) {
    sum += yp[i] * x[i * x_stride];
  }
  return sum;
}
#endif

MAT_INTERNAL_STATIC mat_elem_t mat_svd_dot_strided_scalar_(const Vec *y,
                                                           const Mat *X,
                                                           size_t col) {
  const mat_elem_t *x = &X->data[col];
  size_t x_stride = X->cols;
  size_t n = X->rows;

  mat_elem_t sum = 0;
  for (size_t i = 0; i < n; i++) {
    sum += y->data[i] * x[i * x_stride];
  }
  return sum;
}

// Max column L2 norm for column-major matrix
// Returns max_j ||data[:,j]||_2 where column j is at data[j*col_len]
MAT_INTERNAL_STATIC mat_elem_t mat_max_col_norm_(const mat_elem_t *data,
                                                 size_t col_len,
                                                 size_t n_cols) {
  mat_elem_t max_norm_sq = 0;
  for (size_t j = 0; j < n_cols; j++) {
    Vec col = {
        .rows = col_len, .cols = 1, .data = (mat_elem_t *)&data[j * col_len]};
    mat_elem_t norm_sq = mat_dot(&col, &col);
    if (norm_sq > max_norm_sq)
      max_norm_sq = norm_sq;
  }
  return MAT_SQRT(max_norm_sq);
}

// Shared Jacobi iteration for SVD/pinv
// W is m×n with columns stored contiguously (column j at W->data[j*m], length m)
// V is n×n (accumulates right singular vectors)
// After iteration: W[:,j] = σ_j * u_j, V[:,j] = v_j
// col_stride: stride between columns in W (= m for column-major, = W->cols for row-major)
MAT_INTERNAL_STATIC void mat_svd_jacobi_iter_(mat_elem_t *W_data, size_t m,
                                              mat_elem_t *V_data, size_t n,
                                              size_t col_stride) {
  const int max_sweeps = 30;
  const mat_elem_t tol = MAT_DEFAULT_EPSILON;

  mat_elem_t *col_norms = (mat_elem_t *)MAT_MALLOC(n * sizeof(mat_elem_t));

  // Compute initial column norms (columns are contiguous)
  for (size_t j = 0; j < n; j++) {
    mat_elem_t *col_j = &W_data[j * col_stride];
    mat_elem_t sum = 0;
    for (size_t k = 0; k < m; k++)
      sum += col_j[k] * col_j[k];
    col_norms[j] = sum;
  }

  for (int sweep = 0; sweep < max_sweeps; sweep++) {
    int rotations = 0;

    for (size_t i = 0; i < n - 1; i++) {
      for (size_t j = i + 1; j < n; j++) {
        mat_elem_t a = col_norms[i];
        mat_elem_t b = col_norms[j];

        // Compute dot product of columns i and j
        mat_elem_t *col_i = &W_data[i * col_stride];
        mat_elem_t *col_j = &W_data[j * col_stride];
        mat_elem_t c = 0;
        for (size_t k = 0; k < m; k++)
          c += col_i[k] * col_j[k];

        if (MAT_FABS(c) < tol * MAT_SQRT(a * b + tol))
          continue;

        rotations++;
        mat_elem_t cs, sn;
        mat_svd_jacobi_rotation(a, b, c, &cs, &sn);

#ifdef MAT_HAS_ARM_NEON
        mat_svd_rotate_cols_neon_(W_data, col_stride, i, j, cs, sn);
        mat_svd_rotate_cols_neon_(V_data, n, i, j, cs, sn);
#else
        mat_svd_rotate_cols_scalar_(W_data, col_stride, i, j, cs, sn);
        mat_svd_rotate_cols_scalar_(V_data, n, i, j, cs, sn);
#endif

        mat_elem_t cs2 = cs * cs, sn2 = sn * sn, cs_sn = cs * sn;
        col_norms[i] = cs2 * a + sn2 * b - 2 * cs_sn * c;
        col_norms[j] = sn2 * a + cs2 * b + 2 * cs_sn * c;
      }
    }

    if (rotations == 0)
      break;
  }

  MAT_FREE(col_norms);
}

MATDEF void mat_svd(const Mat *A, Mat *U, Vec *S, Mat *Vt) {
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(U);
  MAT_ASSERT_MAT(S);
  MAT_ASSERT_MAT(Vt);

  size_t m = A->rows;
  size_t n = A->cols;
  size_t k = (m < n) ? m : n; // number of singular values

  MAT_ASSERT(U->rows == m && U->cols == m);
  MAT_ASSERT(S->rows == k && S->cols == 1);
  MAT_ASSERT(Vt->rows == n && Vt->cols == n);

  // Handle m < n by working on A^T
  // SVD(A) = U * S * Vt  =>  SVD(A^T) = V * S * Ut
  if (m < n) {
    Mat *At = mat_rt(A);
    Mat *Ut_temp = mat_mat(n, n);
    Mat *V_temp = mat_mat(m, m);

    mat_svd(At, Ut_temp, S, V_temp);

    // U = V_temp^T, Vt = Ut_temp^T
    mat_t(U, V_temp);
    mat_t(Vt, Ut_temp);

    MAT_FREE_MAT(At);
    MAT_FREE_MAT(Ut_temp);
    MAT_FREE_MAT(V_temp);
    return;
  }

  // Now m >= n
  // W is m×n working matrix: columns will become scaled left singular vectors
  // V is n×n: columns will become right singular vectors
  // For efficient SIMD, we want columns contiguous.
  // Allocate W and V with column-major layout regardless of MAT_COLUMN_MAJOR flag.
  // W_data[col * m + row] = W[row, col], V_data[col * n + row] = V[row, col]
  mat_elem_t *W_data = (mat_elem_t *)MAT_MALLOC(m * n * sizeof(mat_elem_t));
  mat_elem_t *V_data = (mat_elem_t *)MAT_MALLOC(n * n * sizeof(mat_elem_t));

  // Copy A to W (converting to column-contiguous layout)
  for (size_t col = 0; col < n; col++) {
    for (size_t row = 0; row < m; row++) {
      W_data[col * m + row] = MAT_AT(A, row, col);
    }
  }

  // Initialize V to identity (column-contiguous)
  memset(V_data, 0, n * n * sizeof(mat_elem_t));
  for (size_t i = 0; i < n; i++) {
    V_data[i * n + i] = 1;
  }

  // Run Jacobi iteration (columns are contiguous, stride = m for W, n for V)
  mat_svd_jacobi_iter_(W_data, m, V_data, n, m);

  const mat_elem_t tol = MAT_DEFAULT_EPSILON;

  // Extract singular values (column norms of W)
  Perm *order = mat_perm(n);
  Vec *sigma = mat_vec(n);
  MAT_ASSERT(order && sigma);

  for (size_t j = 0; j < n; j++) {
    mat_elem_t *col_j = &W_data[j * m];
    mat_elem_t sum = 0;
    for (size_t i = 0; i < m; i++)
      sum += col_j[i] * col_j[i];
    sigma->data[j] = MAT_SQRT(sum);
    order->data[j] = j;
  }

  // Sort singular values in descending order (simple bubble sort for small n)
  for (size_t i = 0; i < n - 1; i++) {
    for (size_t j = i + 1; j < n; j++) {
      if (sigma->data[order->data[j]] > sigma->data[order->data[i]]) {
        size_t tmp = order->data[i];
        order->data[i] = order->data[j];
        order->data[j] = tmp;
      }
    }
  }

  // Build U: columns are normalized columns of W (in sorted order)
  memset(U->data, 0, m * m * sizeof(mat_elem_t));

  // Track how many columns of U we've properly filled
  size_t u_col = 0;
  for (size_t j = 0; j < n; j++) {
    size_t src = order->data[j];
    mat_elem_t s = sigma->data[src];
    if (s > tol) {
      mat_elem_t *w_col = &W_data[src * m];
      mat_elem_t inv_s = 1 / s;
      for (size_t row = 0; row < m; row++) {
        MAT_SET(U, row, u_col, w_col[row] * inv_s);
      }
      u_col++;
    }
  }

  // Complete U to full orthonormal basis using Modified Gram-Schmidt
  Vec *v = mat_vec(m);

  // First, reorthogonalize the columns we got from W
  for (size_t col = 0; col < u_col; col++) {
    // Copy column to v
    for (size_t i = 0; i < m; i++) {
      v->data[i] = MAT_AT(U, i, col);
    }

    // Orthogonalize against previous columns (twice for stability)
    for (int pass = 0; pass < 2; pass++) {
      for (size_t j = 0; j < col; j++) {
        mat_elem_t dot = 0;
        for (size_t i = 0; i < m; i++)
          dot += v->data[i] * MAT_AT(U, i, j);
        for (size_t i = 0; i < m; i++)
          v->data[i] -= dot * MAT_AT(U, i, j);
      }
    }

    // Normalize and copy back to U
    if (mat_normalize(v) > tol) {
      for (size_t i = 0; i < m; i++) {
        MAT_SET(U, i, col, v->data[i]);
      }
    }
  }

  // Complete the basis with additional columns from standard basis vectors
  for (size_t basis = 0; basis < m && u_col < m; basis++) {
    // Start with e_basis (standard basis vector)
    memset(v->data, 0, m * sizeof(mat_elem_t));
    v->data[basis] = 1;

    // Orthogonalize against all existing columns (twice for stability)
    for (int pass = 0; pass < 2; pass++) {
      for (size_t j = 0; j < u_col; j++) {
        mat_elem_t dot = 0;
        for (size_t i = 0; i < m; i++)
          dot += v->data[i] * MAT_AT(U, i, j);
        for (size_t i = 0; i < m; i++)
          v->data[i] -= dot * MAT_AT(U, i, j);
      }
    }

    // Normalize and add as new column if norm is significant
    if (mat_normalize(v) > tol) {
      for (size_t i = 0; i < m; i++) {
        MAT_SET(U, i, u_col, v->data[i]);
      }
      u_col++;
    }
  }

  // Copy singular values to S (first k = min(m,n) = n since m >= n)
  for (size_t i = 0; i < k; i++) {
    S->data[i] = sigma->data[order->data[i]];
  }

  // Reorthogonalize V (accumulated rotations can drift for large matrices)
  Vec *vv = mat_vec(n);
  for (size_t col = 0; col < n; col++) {
    // Copy column col of V to vv
    mat_elem_t *v_col = &V_data[col * n];
    for (size_t i = 0; i < n; i++)
      vv->data[i] = v_col[i];

    // Orthogonalize against previous columns (twice for stability)
    for (int pass = 0; pass < 2; pass++) {
      for (size_t j = 0; j < col; j++) {
        mat_elem_t *v_col_j = &V_data[j * n];
        mat_elem_t dot = 0;
        for (size_t i = 0; i < n; i++)
          dot += vv->data[i] * v_col_j[i];
        for (size_t i = 0; i < n; i++)
          vv->data[i] -= dot * v_col_j[i];
      }
    }

    // Normalize and copy back
    mat_normalize(vv);
    for (size_t i = 0; i < n; i++)
      v_col[i] = vv->data[i];
  }
  MAT_FREE_MAT(vv);

  MAT_FREE_MAT(v);

  // Build Vt from V with column reordering
  // Vt[i, :] = V[:, order->data[i]]^T
  for (size_t i = 0; i < n; i++) {
    mat_elem_t *v_col = &V_data[order->data[i] * n];
    for (size_t j = 0; j < n; j++) {
      MAT_SET(Vt, i, j, v_col[j]);
    }
  }

  MAT_FREE(W_data);
  MAT_FREE(V_data);
  MAT_FREE_PERM(order);
  MAT_FREE_MAT(sigma);
}

// Moore-Penrose pseudoinverse via thin SVD (Jacobi)
// Computes A+ = V * diag(1/S) * U^T directly from Jacobi iteration
// without building full U basis (much faster for tall/wide matrices)
MATDEF void mat_pinv(Mat *out, const Mat *A) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(A);

  size_t m = A->rows;
  size_t n = A->cols;

  MAT_ASSERT(out->rows == n);
  MAT_ASSERT(out->cols == m);

  // Handle wide matrices by transposing: pinv(A) = pinv(A^T)^T
  if (m < n) {
    Mat *At = mat_rt(A);
    Mat *pinv_At = mat_mat(m, n);
    mat_pinv(pinv_At, At);
    mat_t(out, pinv_At);
    MAT_FREE_MAT(At);
    MAT_FREE_MAT(pinv_At);
    return;
  }

  // Now m >= n
  // Use column-contiguous layout for W and V (same as mat_svd)
  mat_elem_t *W_data = (mat_elem_t *)MAT_MALLOC(m * n * sizeof(mat_elem_t));
  mat_elem_t *V_data = (mat_elem_t *)MAT_MALLOC(n * n * sizeof(mat_elem_t));

  // Copy A to W (column-contiguous)
  for (size_t col = 0; col < n; col++) {
    for (size_t row = 0; row < m; row++) {
      W_data[col * m + row] = MAT_AT(A, row, col);
    }
  }

  // Initialize V to identity (column-contiguous)
  memset(V_data, 0, n * n * sizeof(mat_elem_t));
  for (size_t i = 0; i < n; i++) {
    V_data[i * n + i] = 1;
  }

  // Run Jacobi iteration
  mat_svd_jacobi_iter_(W_data, m, V_data, n, m);

  // Tolerance based on max singular value
  mat_elem_t max_sigma = mat_max_col_norm_(W_data, m, n);
  mat_elem_t pinv_tol =
      (mat_elem_t)(m > n ? m : n) * max_sigma * MAT_DEFAULT_EPSILON;

  // Zero output
  memset(out->data, 0, n * m * sizeof(mat_elem_t));

  // Build pinv via rank-1 updates
  // pinv = Σ (1/σ_j²) * v_j * w_j^T
  for (size_t j = 0; j < n; j++) {
    mat_elem_t *w_col = &W_data[j * m];
    mat_elem_t *v_col = &V_data[j * n];

    // Compute sigma_sq = ||w_col||²
    mat_elem_t sigma_sq = 0;
    for (size_t i = 0; i < m; i++)
      sigma_sq += w_col[i] * w_col[i];

    if (sigma_sq > pinv_tol * pinv_tol) {
      mat_elem_t scale = 1 / sigma_sq;
      // out += scale * v_col * w_col^T (rank-1 update)
      for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < m; k++) {
          MAT_SET(out, i, k, MAT_AT(out, i, k) + scale * v_col[i] * w_col[k]);
        }
      }
    }
  }

  MAT_FREE(W_data);
  MAT_FREE(V_data);
}

MATDEF size_t mat_rank(const Mat *A) {
  MAT_ASSERT_MAT(A);

  size_t m = A->rows;
  size_t n = A->cols;

  if (m == 0 || n == 0)
    return 0;

  size_t k = m < n ? m : n;
  Mat *U = mat_mat(m, m);
  Vec *S = mat_vec(k);
  Mat *Vt = mat_mat(n, n);

  mat_svd(A, U, S, Vt);

  // Tolerance based on max singular value (S[0] is the largest)
  mat_elem_t tol =
      (mat_elem_t)(m > n ? m : n) * S->data[0] * MAT_DEFAULT_EPSILON;

  // Count singular values above tolerance
  size_t rank = 0;
  for (size_t i = 0; i < k; i++) {
    if (S->data[i] > tol)
      rank++;
  }

  MAT_FREE_MAT(U);
  MAT_FREE_MAT(S);
  MAT_FREE_MAT(Vt);
  return rank;
}

MATDEF mat_elem_t mat_cond(const Mat *A) {
  MAT_ASSERT_MAT(A);

  size_t m = A->rows;
  size_t n = A->cols;

  if (m == 0 || n == 0)
    return 0;

  size_t k = m < n ? m : n;
  Mat *U = mat_mat(m, m);
  Vec *S = mat_vec(k);
  Mat *Vt = mat_mat(n, n);

  mat_svd(A, U, S, Vt);

  // S[0] is max, S[k-1] is min (singular values are sorted descending)
  mat_elem_t sigma_max = S->data[0];
  mat_elem_t sigma_min = S->data[k - 1];

  MAT_FREE_MAT(U);
  MAT_FREE_MAT(S);
  MAT_FREE_MAT(Vt);

  // Handle singular matrices
  if (sigma_min < MAT_DEFAULT_EPSILON * sigma_max) {
    return MAT_HUGE_VAL;
  }

  return sigma_max / sigma_min;
}

#endif // MAT_IMPLEMENTATION
