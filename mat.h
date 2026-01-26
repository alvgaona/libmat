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

// x86 AVX2 detection
#if defined(__AVX2__)
#define MAT_HAS_AVX2
#include <immintrin.h>
#endif

// Architecture indicator macro - shows which SIMD backend is active
// Usage: printf("libmat using: %s\n", MAT_ARCH);
#if defined(MAT_HAS_ARM_NEON)
#define MAT_ARCH "NEON"
#elif defined(MAT_HAS_AVX2)
#define MAT_ARCH "AVX2"
#else
#define MAT_ARCH "SCALAR"
#endif

/* SIMD Dispatch Architecture */

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

/* AVX2 SIMD Macros (TODO) */
#ifdef MAT_HAS_AVX2
// AVX2 double precision (4 doubles per 256-bit register)
// AVX2 single precision (8 floats per 256-bit register)
// TODO: Add AVX2 abstraction macros similar to NEON
// When implementing, use _mm256_* intrinsics from <immintrin.h>
#endif // MAT_HAS_AVX2

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

// Logging (set MAT_LOG_LEVEL > 0 to enable)
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

// Assertions (define MAT_ASSERT to override)
#ifndef MAT_ASSERT
#include <assert.h>
#define MAT_ASSERT(x) assert(x)
#endif

// Type-specific assertion macros
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

// Matrix dimensions tuple
typedef struct {
  size_t x;
  size_t y;
} MatSize;

// Vectors in this library are implemented as a Mat
// of size nx1 (column vector) by default.
// Row vectors are just transposed column vectors.
typedef Mat Vec;

// Permutation vector (array of indices)
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

// C = alpha * A * B^T + alpha * B * A^T + beta * C (SYR2K). SIMD-optimized.
// Symmetric rank-2k update. uplo: 'L' for lower, 'U' for upper triangle.
MATDEF void mat_syr2k(Mat *C, const Mat *A, const Mat *B, mat_elem_t alpha,
                      mat_elem_t beta, char uplo);

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
// Uses Hessenberg reduction followed by implicit QR iteration.
// Eigenvalues are returned in arbitrary order.
// For complex eigenvalues (conjugate pairs), the real part is stored.
MATDEF void mat_eigvals(Vec *out, const Mat *A);

// Eigenvalues of symmetric matrix (faster than mat_eigvals for symmetric input).
// Uses tridiagonal reduction + implicit QR iteration. O(n) per QR step vs O(n^2).
// Eigenvalues are returned in arbitrary order.
MATDEF void mat_eigvals_sym(Vec *out, const Mat *A);

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

/* BLAS-like Kernels */

/* Kernel dispatch macro - selects architecture-specific implementation */
#ifdef MAT_HAS_ARM_NEON
#define MAT_DISPATCH(name) mat__##name##_neon
#elif defined(MAT_HAS_AVX2)
#define MAT_DISPATCH(name) mat__##name##_avx2
#else
#define MAT_DISPATCH(name) mat__##name##_scalar
#endif

// AXPY: y[0:n] += alpha * x[0:n]
MAT_INTERNAL_STATIC void mat__axpy_scalar(mat_elem_t *y, mat_elem_t alpha,
                                          const mat_elem_t *x, size_t n) {
  for (size_t i = 0; i < n; i++) {
    y[i] += alpha * x[i];
  }
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void mat__axpy_neon(mat_elem_t *y, mat_elem_t alpha,
                                        const mat_elem_t *x, size_t n) {
  MAT_NEON_TYPE valpha = MAT_NEON_DUP(alpha);

  size_t i = 0;
  for (; i + MAT_NEON_WIDTH * 4 <= n; i += MAT_NEON_WIDTH * 4) {
    MAT_NEON_TYPE vy0 = MAT_NEON_LOAD(&y[i]);
    MAT_NEON_TYPE vy1 = MAT_NEON_LOAD(&y[i + MAT_NEON_WIDTH]);
    MAT_NEON_TYPE vy2 = MAT_NEON_LOAD(&y[i + MAT_NEON_WIDTH * 2]);
    MAT_NEON_TYPE vy3 = MAT_NEON_LOAD(&y[i + MAT_NEON_WIDTH * 3]);
    MAT_NEON_TYPE vx0 = MAT_NEON_LOAD(&x[i]);
    MAT_NEON_TYPE vx1 = MAT_NEON_LOAD(&x[i + MAT_NEON_WIDTH]);
    MAT_NEON_TYPE vx2 = MAT_NEON_LOAD(&x[i + MAT_NEON_WIDTH * 2]);
    MAT_NEON_TYPE vx3 = MAT_NEON_LOAD(&x[i + MAT_NEON_WIDTH * 3]);
    vy0 = MAT_NEON_FMA(vy0, vx0, valpha);
    vy1 = MAT_NEON_FMA(vy1, vx1, valpha);
    vy2 = MAT_NEON_FMA(vy2, vx2, valpha);
    vy3 = MAT_NEON_FMA(vy3, vx3, valpha);
    MAT_NEON_STORE(&y[i], vy0);
    MAT_NEON_STORE(&y[i + MAT_NEON_WIDTH], vy1);
    MAT_NEON_STORE(&y[i + MAT_NEON_WIDTH * 2], vy2);
    MAT_NEON_STORE(&y[i + MAT_NEON_WIDTH * 3], vy3);
  }

  for (; i + MAT_NEON_WIDTH <= n; i += MAT_NEON_WIDTH) {
    MAT_NEON_TYPE vy = MAT_NEON_LOAD(&y[i]);
    MAT_NEON_TYPE vx = MAT_NEON_LOAD(&x[i]);
    vy = MAT_NEON_FMA(vy, vx, valpha);
    MAT_NEON_STORE(&y[i], vy);
  }

  for (; i < n; i++) {
    y[i] += alpha * x[i];
  }
}
#endif

MAT_INTERNAL_STATIC void mat__axpy_kernel(mat_elem_t *y, mat_elem_t alpha,
                                          const mat_elem_t *x, size_t n) {
  MAT_DISPATCH(axpy)(y, alpha, x, n);
}

// SCAL: y[0:n] *= alpha
MAT_INTERNAL_STATIC void mat__scal_scalar(mat_elem_t *y, mat_elem_t alpha, size_t n) {
  for (size_t i = 0; i < n; i++) {
    y[i] *= alpha;
  }
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void mat__scal_neon(mat_elem_t *y, mat_elem_t alpha, size_t n) {
  MAT_NEON_TYPE valpha = MAT_NEON_DUP(alpha);

  size_t i = 0;
  for (; i + MAT_NEON_WIDTH * 4 <= n; i += MAT_NEON_WIDTH * 4) {
    MAT_NEON_TYPE vy0 = MAT_NEON_LOAD(&y[i]);
    MAT_NEON_TYPE vy1 = MAT_NEON_LOAD(&y[i + MAT_NEON_WIDTH]);
    MAT_NEON_TYPE vy2 = MAT_NEON_LOAD(&y[i + MAT_NEON_WIDTH * 2]);
    MAT_NEON_TYPE vy3 = MAT_NEON_LOAD(&y[i + MAT_NEON_WIDTH * 3]);
    MAT_NEON_STORE(&y[i], MAT_NEON_MUL(vy0, valpha));
    MAT_NEON_STORE(&y[i + MAT_NEON_WIDTH], MAT_NEON_MUL(vy1, valpha));
    MAT_NEON_STORE(&y[i + MAT_NEON_WIDTH * 2], MAT_NEON_MUL(vy2, valpha));
    MAT_NEON_STORE(&y[i + MAT_NEON_WIDTH * 3], MAT_NEON_MUL(vy3, valpha));
  }

  for (; i + MAT_NEON_WIDTH <= n; i += MAT_NEON_WIDTH) {
    MAT_NEON_TYPE vy = MAT_NEON_LOAD(&y[i]);
    MAT_NEON_STORE(&y[i], MAT_NEON_MUL(vy, valpha));
  }

  for (; i < n; i++) {
    y[i] *= alpha;
  }
}
#endif

MAT_INTERNAL_STATIC void mat__scal_kernel(mat_elem_t *y, mat_elem_t alpha, size_t n) {
  if (alpha == 1) return;
  if (alpha == 0) {
    memset(y, 0, n * sizeof(mat_elem_t));
    return;
  }
  MAT_DISPATCH(scal)(y, alpha, n);
}

// DOT: result = sum(a[i] * b[i])
MAT_INTERNAL_STATIC mat_elem_t mat__dot_scalar(const mat_elem_t *a,
                                               const mat_elem_t *b, size_t n) {
  mat_elem_t result = 0;
  for (size_t i = 0; i < n; i++) {
    result += a[i] * b[i];
  }
  return result;
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC mat_elem_t mat__dot_neon(const mat_elem_t *a,
                                             const mat_elem_t *b, size_t n) {
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
}
#endif

MAT_INTERNAL_STATIC mat_elem_t mat__dot_kernel(const mat_elem_t *a,
                                               const mat_elem_t *b, size_t n) {
  return MAT_DISPATCH(dot)(a, b, n);
}

// SYR2_COL: y[i] -= alpha*a[i] + beta*b[i]
MAT_INTERNAL_STATIC void mat__syr2_col_scalar(mat_elem_t *y, mat_elem_t alpha,
                                              const mat_elem_t *a, mat_elem_t beta,
                                              const mat_elem_t *b, size_t n) {
  for (size_t i = 0; i < n; i++) {
    y[i] -= alpha * a[i] + beta * b[i];
  }
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void mat__syr2_col_neon(mat_elem_t *y, mat_elem_t alpha,
                                            const mat_elem_t *a, mat_elem_t beta,
                                            const mat_elem_t *b, size_t n) {
  MAT_NEON_TYPE valpha = MAT_NEON_DUP(alpha);
  MAT_NEON_TYPE vbeta = MAT_NEON_DUP(beta);

  size_t i = 0;
  for (; i + MAT_NEON_WIDTH * 4 <= n; i += MAT_NEON_WIDTH * 4) {
    MAT_NEON_TYPE vy0 = MAT_NEON_LOAD(&y[i]);
    MAT_NEON_TYPE vy1 = MAT_NEON_LOAD(&y[i + MAT_NEON_WIDTH]);
    MAT_NEON_TYPE vy2 = MAT_NEON_LOAD(&y[i + MAT_NEON_WIDTH * 2]);
    MAT_NEON_TYPE vy3 = MAT_NEON_LOAD(&y[i + MAT_NEON_WIDTH * 3]);

    MAT_NEON_TYPE va0 = MAT_NEON_LOAD(&a[i]);
    MAT_NEON_TYPE va1 = MAT_NEON_LOAD(&a[i + MAT_NEON_WIDTH]);
    MAT_NEON_TYPE va2 = MAT_NEON_LOAD(&a[i + MAT_NEON_WIDTH * 2]);
    MAT_NEON_TYPE va3 = MAT_NEON_LOAD(&a[i + MAT_NEON_WIDTH * 3]);

    MAT_NEON_TYPE vb0 = MAT_NEON_LOAD(&b[i]);
    MAT_NEON_TYPE vb1 = MAT_NEON_LOAD(&b[i + MAT_NEON_WIDTH]);
    MAT_NEON_TYPE vb2 = MAT_NEON_LOAD(&b[i + MAT_NEON_WIDTH * 2]);
    MAT_NEON_TYPE vb3 = MAT_NEON_LOAD(&b[i + MAT_NEON_WIDTH * 3]);

    // y -= alpha*a + beta*b
    vy0 = MAT_NEON_SUB(vy0, MAT_NEON_FMA(MAT_NEON_MUL(valpha, va0), vb0, vbeta));
    vy1 = MAT_NEON_SUB(vy1, MAT_NEON_FMA(MAT_NEON_MUL(valpha, va1), vb1, vbeta));
    vy2 = MAT_NEON_SUB(vy2, MAT_NEON_FMA(MAT_NEON_MUL(valpha, va2), vb2, vbeta));
    vy3 = MAT_NEON_SUB(vy3, MAT_NEON_FMA(MAT_NEON_MUL(valpha, va3), vb3, vbeta));

    MAT_NEON_STORE(&y[i], vy0);
    MAT_NEON_STORE(&y[i + MAT_NEON_WIDTH], vy1);
    MAT_NEON_STORE(&y[i + MAT_NEON_WIDTH * 2], vy2);
    MAT_NEON_STORE(&y[i + MAT_NEON_WIDTH * 3], vy3);
  }

  for (; i + MAT_NEON_WIDTH <= n; i += MAT_NEON_WIDTH) {
    MAT_NEON_TYPE vy = MAT_NEON_LOAD(&y[i]);
    MAT_NEON_TYPE va = MAT_NEON_LOAD(&a[i]);
    MAT_NEON_TYPE vb = MAT_NEON_LOAD(&b[i]);
    vy = MAT_NEON_SUB(vy, MAT_NEON_FMA(MAT_NEON_MUL(valpha, va), vb, vbeta));
    MAT_NEON_STORE(&y[i], vy);
  }

  for (; i < n; i++) {
    y[i] -= alpha * a[i] + beta * b[i];
  }
}
#endif

MAT_INTERNAL_STATIC void mat__syr2_col_kernel(mat_elem_t *y, mat_elem_t alpha,
                                              const mat_elem_t *a, mat_elem_t beta,
                                              const mat_elem_t *b, size_t n) {
  MAT_DISPATCH(syr2_col)(y, alpha, a, beta, b, n);
}

// COPY: dest[0:n] = src[0:n]
MAT_INTERNAL_STATIC void mat__copy_scalar(mat_elem_t *dest, const mat_elem_t *src,
                                          size_t n) {
  memcpy(dest, src, n * sizeof(mat_elem_t));
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void mat__copy_neon(mat_elem_t *dest, const mat_elem_t *src,
                                        size_t n) {
  size_t i = 0;
  for (; i + MAT_NEON_WIDTH * 4 <= n; i += MAT_NEON_WIDTH * 4) {
    MAT_NEON_STORE(&dest[i], MAT_NEON_LOAD(&src[i]));
    MAT_NEON_STORE(&dest[i + MAT_NEON_WIDTH], MAT_NEON_LOAD(&src[i + MAT_NEON_WIDTH]));
    MAT_NEON_STORE(&dest[i + MAT_NEON_WIDTH * 2], MAT_NEON_LOAD(&src[i + MAT_NEON_WIDTH * 2]));
    MAT_NEON_STORE(&dest[i + MAT_NEON_WIDTH * 3], MAT_NEON_LOAD(&src[i + MAT_NEON_WIDTH * 3]));
  }
  for (; i < n; i++) {
    dest[i] = src[i];
  }
}
#endif

MAT_INTERNAL_STATIC void mat__copy_kernel(mat_elem_t *dest, const mat_elem_t *src,
                                          size_t n) {
  MAT_DISPATCH(copy)(dest, src, n);
}

// SWAP: swap a[0:n] <-> b[0:n]
MAT_INTERNAL_STATIC void mat__swap_scalar(mat_elem_t *a, mat_elem_t *b, size_t n) {
  for (size_t i = 0; i < n; i++) {
    mat_elem_t tmp = a[i];
    a[i] = b[i];
    b[i] = tmp;
  }
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void mat__swap_neon(mat_elem_t *a, mat_elem_t *b, size_t n) {
  size_t i = 0;
  for (; i + MAT_NEON_WIDTH * 4 <= n; i += MAT_NEON_WIDTH * 4) {
    MAT_NEON_TYPE a0 = MAT_NEON_LOAD(&a[i]);
    MAT_NEON_TYPE a1 = MAT_NEON_LOAD(&a[i + MAT_NEON_WIDTH]);
    MAT_NEON_TYPE a2 = MAT_NEON_LOAD(&a[i + MAT_NEON_WIDTH * 2]);
    MAT_NEON_TYPE a3 = MAT_NEON_LOAD(&a[i + MAT_NEON_WIDTH * 3]);
    MAT_NEON_TYPE b0 = MAT_NEON_LOAD(&b[i]);
    MAT_NEON_TYPE b1 = MAT_NEON_LOAD(&b[i + MAT_NEON_WIDTH]);
    MAT_NEON_TYPE b2 = MAT_NEON_LOAD(&b[i + MAT_NEON_WIDTH * 2]);
    MAT_NEON_TYPE b3 = MAT_NEON_LOAD(&b[i + MAT_NEON_WIDTH * 3]);
    MAT_NEON_STORE(&a[i], b0);
    MAT_NEON_STORE(&a[i + MAT_NEON_WIDTH], b1);
    MAT_NEON_STORE(&a[i + MAT_NEON_WIDTH * 2], b2);
    MAT_NEON_STORE(&a[i + MAT_NEON_WIDTH * 3], b3);
    MAT_NEON_STORE(&b[i], a0);
    MAT_NEON_STORE(&b[i + MAT_NEON_WIDTH], a1);
    MAT_NEON_STORE(&b[i + MAT_NEON_WIDTH * 2], a2);
    MAT_NEON_STORE(&b[i + MAT_NEON_WIDTH * 3], a3);
  }
  for (; i < n; i++) {
    mat_elem_t tmp = a[i];
    a[i] = b[i];
    b[i] = tmp;
  }
}
#endif

MAT_INTERNAL_STATIC void mat__swap_kernel(mat_elem_t *a, mat_elem_t *b, size_t n) {
  MAT_DISPATCH(swap)(a, b, n);
}

// AMAX: returns max(|arr[0:n]|) (BLAS AMAX - absolute max value)
MAT_INTERNAL_STATIC mat_elem_t mat__amax_scalar(const mat_elem_t *arr, size_t n) {
  if (n == 0) return 0;
  mat_elem_t max_val = MAT_FABS(arr[0]);
  for (size_t i = 1; i < n; i++) {
    mat_elem_t val = MAT_FABS(arr[i]);
    if (val > max_val) max_val = val;
  }
  return max_val;
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC mat_elem_t mat__amax_neon(const mat_elem_t *arr, size_t n) {
  if (n == 0) return 0;
  MAT_NEON_TYPE vmax0 = MAT_NEON_DUP(0);
  MAT_NEON_TYPE vmax1 = MAT_NEON_DUP(0);
  MAT_NEON_TYPE vmax2 = MAT_NEON_DUP(0);
  MAT_NEON_TYPE vmax3 = MAT_NEON_DUP(0);
  size_t i = 0;
  for (; i + MAT_NEON_WIDTH * 4 <= n; i += MAT_NEON_WIDTH * 4) {
    MAT_NEON_TYPE v0 = MAT_NEON_ABS(MAT_NEON_LOAD(&arr[i]));
    MAT_NEON_TYPE v1 = MAT_NEON_ABS(MAT_NEON_LOAD(&arr[i + MAT_NEON_WIDTH]));
    MAT_NEON_TYPE v2 = MAT_NEON_ABS(MAT_NEON_LOAD(&arr[i + MAT_NEON_WIDTH * 2]));
    MAT_NEON_TYPE v3 = MAT_NEON_ABS(MAT_NEON_LOAD(&arr[i + MAT_NEON_WIDTH * 3]));
    vmax0 = MAT_NEON_MAX(vmax0, v0);
    vmax1 = MAT_NEON_MAX(vmax1, v1);
    vmax2 = MAT_NEON_MAX(vmax2, v2);
    vmax3 = MAT_NEON_MAX(vmax3, v3);
  }
  for (; i + MAT_NEON_WIDTH <= n; i += MAT_NEON_WIDTH) {
    MAT_NEON_TYPE v = MAT_NEON_ABS(MAT_NEON_LOAD(&arr[i]));
    vmax0 = MAT_NEON_MAX(vmax0, v);
  }
  vmax0 = MAT_NEON_MAX(vmax0, vmax1);
  vmax2 = MAT_NEON_MAX(vmax2, vmax3);
  vmax0 = MAT_NEON_MAX(vmax0, vmax2);
  mat_elem_t max_val = MAT_NEON_MAXV(vmax0);
  // Scalar remainder
  for (; i < n; i++) {
    mat_elem_t val = MAT_FABS(arr[i]);
    if (val > max_val) max_val = val;
  }
  return max_val;
}
#endif

MAT_INTERNAL_STATIC mat_elem_t mat__amax_kernel(const mat_elem_t *arr, size_t n) {
  return MAT_DISPATCH(amax)(arr, n);
}

// IAMAX: returns index of max(|arr[0:n]|) (BLAS IAMAX)
// Also returns the max value via max_out if non-NULL
MAT_INTERNAL_STATIC size_t mat__iamax(const mat_elem_t *arr, size_t n,
                                          mat_elem_t *max_out) {
  if (n == 0) {
    if (max_out) *max_out = 0;
    return 0;
  }

  mat_elem_t max_val = mat__amax_kernel(arr, n);
  if (max_out) *max_out = max_val;

  // Find index of max value
  for (size_t i = 0; i < n; i++) {
    if (MAT_FABS(arr[i]) == max_val) return i;
  }
  return 0;  // Fallback (shouldn't happen)
}

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
MAT_INTERNAL_STATIC bool mat__equals_tol_neon_(const Mat *a, const Mat *b,
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

MAT_INTERNAL_STATIC bool mat__equals_tol_scalar_(const Mat *a, const Mat *b,
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

MAT_INTERNAL_STATIC bool mat__equals_tol_dispatch_(const Mat *a, const Mat *b,
                                                   mat_elem_t epsilon) {
#if defined(MAT_HAS_ARM_NEON)
  return mat__equals_tol_neon_(a, b, epsilon);
#elif defined(MAT_HAS_AVX2)
  return mat_equals_tol_avx2_(a, b, epsilon);  // Future
#else
  return mat__equals_tol_scalar_(a, b, epsilon);
#endif
}

MATDEF bool mat_equals_tol(const Mat *a, const Mat *b, mat_elem_t epsilon) {
  MAT_ASSERT_MAT(a);
  MAT_ASSERT_MAT(b);

  if (a->rows != b->rows || a->cols != b->cols)
    return false;

  return mat__equals_tol_dispatch_(a, b, epsilon);
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
  mat__scal_kernel(out->data, k, out->rows * out->cols);
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

MATDEF mat_elem_t mat_dot(const Vec *v1, const Vec *v2) {
  MAT_ASSERT_MAT(v1);
  MAT_ASSERT_MAT(v2);
  MAT_ASSERT(v1->rows * v1->cols == v2->rows * v2->cols);
  return mat__dot_kernel(v1->data, v2->data, v1->rows * v1->cols);
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
MAT_INTERNAL_STATIC void mat__outer_neon_(Mat *out, const Vec *v1,
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

MAT_INTERNAL_STATIC void mat__outer_scalar_(Mat *out, const Vec *v1,
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

MAT_INTERNAL_STATIC void mat__outer_dispatch_(Mat *out, const Vec *v1,
                                              const Vec *v2) {
#if defined(MAT_HAS_ARM_NEON)
  mat__outer_neon_(out, v1, v2);
#elif defined(MAT_HAS_AVX2)
  mat_outer_avx2_(out, v1, v2);  // Future
#else
  mat__outer_scalar_(out, v1, v2);
#endif
}

MATDEF void mat_outer(Mat *out, const Vec *v1, const Vec *v2) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(v1);
  MAT_ASSERT_MAT(v2);

  size_t m = v1->rows * v1->cols;
  size_t n = v2->rows * v2->cols;
  MAT_ASSERT(out->rows == m && out->cols == n);

  mat__outer_dispatch_(out, v1, v2);
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

/* Fused Operations */

// y += alpha * x (BLAS Level-1: axpy)
MATDEF void mat_axpy(Vec *y, mat_elem_t alpha, const Vec *x) {
  MAT_ASSERT_MAT(y);
  MAT_ASSERT_MAT(x);
  MAT_ASSERT(y->rows == x->rows);

  mat__axpy_kernel(y->data, alpha, x->data, x->rows);
}

// y = alpha * A * x + beta * y (BLAS Level-2: gemv)
#ifdef MAT_HAS_ARM_NEON
// Forward declaration for fallback
MAT_INTERNAL_STATIC void mat__gemv_scalar_(Vec *y, mat_elem_t alpha,
                                          const Mat *A, const Vec *x,
                                          mat_elem_t beta);

MAT_INTERNAL_STATIC void mat__gemv_neon_(Vec *y, mat_elem_t alpha, const Mat *A,
                                        const Vec *x, mat_elem_t beta) {
  // Column-major: columns are contiguous, use AXPY-style algorithm
  // Process 8 columns at a time for better instruction-level parallelism
  size_t m = A->rows;
  size_t n = A->cols;
  mat_elem_t *py = y->data;
  const mat_elem_t *pa = A->data;
  const mat_elem_t *px = x->data;

  // Scale y by beta first
  mat__scal_kernel(py, beta, m);

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

MAT_INTERNAL_STATIC void mat__gemv_scalar_(Vec *y, mat_elem_t alpha,
                                          const Mat *A, const Vec *x,
                                          mat_elem_t beta) {
  size_t m = A->rows;
  size_t n = A->cols;
  mat_elem_t *py = y->data;
  const mat_elem_t *pa = A->data;
  const mat_elem_t *px = x->data;

  // Scale y by beta, then accumulate column-by-column: y += alpha * x[j] * A[:,j]
  mat__scal_kernel(py, beta, m);
  for (size_t j = 0; j < n; j++) {
    mat__axpy_kernel(py, alpha * px[j], &pa[j * m], m);
  }
}

MAT_INTERNAL_STATIC void mat__gemv_dispatch_(Vec *y, mat_elem_t alpha,
                                             const Mat *A, const Vec *x,
                                             mat_elem_t beta) {
#if defined(MAT_HAS_ARM_NEON)
  mat__gemv_neon_(y, alpha, A, x, beta);
#elif defined(MAT_HAS_AVX2)
  mat_gemv_avx2_(y, alpha, A, x, beta);  // Future
#else
  mat__gemv_scalar_(y, alpha, A, x, beta);
#endif
}

MATDEF void mat_gemv(Vec *y, mat_elem_t alpha, const Mat *A, const Vec *x,
                     mat_elem_t beta) {
  MAT_ASSERT_MAT(y);
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(x);
  MAT_ASSERT(A->rows == y->rows);
  MAT_ASSERT(A->cols == x->rows);

  mat__gemv_dispatch_(y, alpha, A, x, beta);
}

// y = alpha * A^T * x + beta * y (GEMV transposed)
// A is m×n, x is m×1, y is n×1
#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void mat__gemv_t_neon_(Vec *y, mat_elem_t alpha,
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

MAT_INTERNAL_STATIC void mat__gemv_t_scalar_(Vec *y, mat_elem_t alpha,
                                            const Mat *A, const Vec *x,
                                            mat_elem_t beta) {
  size_t m = A->rows;
  size_t n = A->cols;
  mat_elem_t *py = y->data;
  const mat_elem_t *pa = A->data;
  const mat_elem_t *px = x->data;

  // y[j] = beta * y[j] + alpha * dot(A[:,j], x)
  for (size_t j = 0; j < n; j++) {
    py[j] = beta * py[j] + alpha * mat__dot_kernel(&pa[j * m], px, m);
  }
}

MAT_INTERNAL_STATIC void mat__gemv_t_dispatch_(Vec *y, mat_elem_t alpha,
                                               const Mat *A, const Vec *x,
                                               mat_elem_t beta) {
#if defined(MAT_HAS_ARM_NEON)
  mat__gemv_t_neon_(y, alpha, A, x, beta);
#elif defined(MAT_HAS_AVX2)
  mat_gemv_t_avx2_(y, alpha, A, x, beta);  // Future
#else
  mat__gemv_t_scalar_(y, alpha, A, x, beta);
#endif
}

MATDEF void mat_gemv_t(Vec *y, mat_elem_t alpha, const Mat *A, const Vec *x,
                       mat_elem_t beta) {
  MAT_ASSERT_MAT(y);
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(x);
  MAT_ASSERT(A->cols == y->rows); // A^T has n rows
  MAT_ASSERT(A->rows == x->rows); // A^T has m cols

  mat__gemv_t_dispatch_(y, alpha, A, x, beta);
}

// A += alpha * x * y^T (BLAS Level-2: ger - rank-1 update)
// Column-major: A[:,j] += alpha * y[j] * x (columns are contiguous)
MAT_INTERNAL_STATIC void mat__ger_(Mat *A, mat_elem_t alpha, const Vec *x,
                                  const Vec *y) {
  size_t m = A->rows;
  size_t n = A->cols;
  mat_elem_t *pa = A->data;
  const mat_elem_t *px = x->data;
  const mat_elem_t *py = y->data;

  for (size_t j = 0; j < n; j++) {
    mat__axpy_kernel(&pa[j * m], alpha * py[j], px, m);
  }
}

MATDEF void mat_ger(Mat *A, mat_elem_t alpha, const Vec *x, const Vec *y) {
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(x);
  MAT_ASSERT_MAT(y);
  MAT_ASSERT(A->rows == x->rows);
  MAT_ASSERT(A->cols == y->rows);

  mat__ger_(A, alpha, x, y);
}

// A += alpha * x * x^T (BLAS Level-2: syr - symmetric rank-1 update)
// Column-major: A[j:n, j] += alpha * x[j] * x[j:n] (lower)
//               A[0:j+1, j] += alpha * x[j] * x[0:j+1] (upper)
MAT_INTERNAL_STATIC void mat__syr_lower_(Mat *A, mat_elem_t alpha,
                                        const Vec *x) {
  size_t n = A->rows;
  mat_elem_t *pa = A->data;
  const mat_elem_t *px = x->data;

  for (size_t j = 0; j < n; j++) {
    mat__axpy_kernel(&pa[j * n + j], alpha * px[j], &px[j], n - j);
  }
}

MAT_INTERNAL_STATIC void mat__syr_upper_(Mat *A, mat_elem_t alpha,
                                        const Vec *x) {
  size_t n = A->rows;
  mat_elem_t *pa = A->data;
  const mat_elem_t *px = x->data;

  for (size_t j = 0; j < n; j++) {
    mat__axpy_kernel(&pa[j * n], alpha * px[j], px, j + 1);
  }
}

MATDEF void mat_syr(Mat *A, mat_elem_t alpha, const Vec *x, char uplo) {
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(x);
  MAT_ASSERT(A->rows == A->cols);
  MAT_ASSERT(A->rows == x->rows);

  if (uplo == 'L' || uplo == 'l') {
    mat__syr_lower_(A, alpha, x);
  } else {
    mat__syr_upper_(A, alpha, x);
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
mat__gemm_unit_lower_t_neon_(mat_elem_t *W, size_t ldw, const mat_elem_t *V,
                            size_t ldv, const mat_elem_t *R, size_t ldr,
                            size_t kb, size_t panel_rows, size_t N) {

  for (size_t ii = 0; ii < kb; ii++) {
    mat_elem_t *Wi = &W[ii * ldw];
    const mat_elem_t *Rii = &R[ii * ldr];

    // Initialize W[ii,:] = R[ii,:] (the diagonal 1 in V^T)
    mat__copy_kernel(Wi, Rii, N);

    // Accumulate: W[ii,:] += V[ii,r] * R[r,:] for r = ii+1 to panel_rows-1
    for (size_t r = ii + 1; r < panel_rows; r++) {
      mat__axpy_kernel(Wi, V[ii * ldv + r], &R[r * ldr], N);
    }
  }
}

// C -= V * W where V is unit lower triangular (panel_rows x kb), W is (kb x N)
// C is (panel_rows x N) with stride ldc
// V stored column-major: V[col * ldv + row] for row > col
MAT_INTERNAL_STATIC void
mat__gemm_unit_lower_neon_(mat_elem_t *C, size_t ldc, const mat_elem_t *V,
                          size_t ldv, const mat_elem_t *W, size_t ldw,
                          size_t panel_rows, size_t kb, size_t N) {

  for (size_t r = 0; r < panel_rows; r++) {
    mat_elem_t *Cr = &C[r * ldc];
    size_t ii_max = (r < kb) ? r + 1 : kb;

    for (size_t ii = 0; ii < ii_max; ii++) {
      mat_elem_t v_val = (r == ii) ? 1.0f : V[ii * ldv + r];
      mat__axpy_kernel(Cr, -v_val, &W[ii * ldw], N);
    }
  }
}

// W = Q * V where Q is (M x panel_rows) with stride ldq, V is unit lower
// (panel_rows x kb) Result W is (M x kb) with stride ldw
MAT_INTERNAL_STATIC void
mat__gemm_q_unit_lower_neon_(mat_elem_t *W, size_t ldw, const mat_elem_t *Q,
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
mat__gemm_sub_w_unit_upper_neon_(mat_elem_t *Q, size_t ldq, const mat_elem_t *W,
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
mat__gemm_unit_lower_t_scalar_(mat_elem_t *W, size_t ldw, const mat_elem_t *V,
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
mat__gemm_unit_lower_scalar_(mat_elem_t *C, size_t ldc, const mat_elem_t *V,
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
mat__gemm_q_unit_lower_scalar_(mat_elem_t *W, size_t ldw, const mat_elem_t *Q,
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

MAT_INTERNAL_STATIC void mat__gemm_sub_w_unit_upper_scalar_(
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
mat__gemm_strided_neon_(mat_elem_t *C, size_t ldc, mat_elem_t alpha,
                                const mat_elem_t *A, size_t lda,
                                mat_trans_t transA, const mat_elem_t *B,
                                size_t ldb, mat_trans_t transB, size_t M,
                                size_t K, size_t N, mat_elem_t beta) {
  // Scale C by beta first
  for (size_t j = 0; j < N; j++)
    mat__scal_kernel(&C[j * ldc], beta, M);

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
mat__gemm_strided_(mat_elem_t *C, size_t ldc, mat_elem_t alpha,
                   const mat_elem_t *A, size_t lda, mat_trans_t transA,
                   const mat_elem_t *B, size_t ldb, mat_trans_t transB,
                   size_t M, size_t K, size_t N, mat_elem_t beta) {
#ifdef MAT_HAS_ARM_NEON
  mat__gemm_strided_neon_(C, ldc, alpha, A, lda, transA, B, ldb, transB,
                          M, K, N, beta);
#else
  // Scalar fallback: scale C by beta first
  for (size_t j = 0; j < N; j++)
    mat__scal_kernel(&C[j * ldc], beta, M);
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
MAT_INTERNAL_STATIC void mat__gemm_scalar_(Mat *C, mat_elem_t alpha,
                                          const Mat *A, const Mat *B,
                                          mat_elem_t beta);

MAT_INTERNAL_STATIC void mat__gemm_neon_(Mat *C, mat_elem_t alpha, const Mat *A,
                                        const Mat *B, mat_elem_t beta) {
  // Column-major: delegate to strided implementation
  // For column-major Mat, leading dimension = rows
  mat__gemm_strided_neon_(C->data, C->rows, alpha, A->data, A->rows,
                                  MAT_NO_TRANS, B->data, B->rows, MAT_NO_TRANS,
                                  A->rows, A->cols, B->cols, beta);
}
#endif // MAT_HAS_ARM_NEON

MAT_INTERNAL_STATIC void mat__gemm_scalar_(Mat *C, mat_elem_t alpha,
                                          const Mat *A, const Mat *B,
                                          mat_elem_t beta) {
  size_t M = A->rows;
  size_t K = A->cols;
  size_t N = B->cols;

  // Scale C by beta first
  mat__scal_kernel(C->data, beta, M * N);

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

MAT_INTERNAL_STATIC void mat__gemm_dispatch_(Mat *C, mat_elem_t alpha,
                                             const Mat *A, const Mat *B,
                                             mat_elem_t beta) {
#if defined(MAT_HAS_ARM_NEON)
  mat__gemm_neon_(C, alpha, A, B, beta);
#elif defined(MAT_HAS_AVX2)
  mat_gemm_avx2_(C, alpha, A, B, beta);  // Future
#else
  mat__gemm_scalar_(C, alpha, A, B, beta);
#endif
}

MATDEF void mat_gemm(Mat *C, mat_elem_t alpha, const Mat *A, const Mat *B,
                     mat_elem_t beta) {
  MAT_ASSERT_MAT(C);
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(B);
  MAT_ASSERT(A->cols == B->rows);
  MAT_ASSERT(C->rows == A->rows && C->cols == B->cols);

  mat__gemm_dispatch_(C, alpha, A, B, beta);
}

/* Structure Operations */

#define MAT_T_BLOCK 32

#ifdef MAT_HAS_ARM_NEON
// Forward declaration for fallback
MAT_INTERNAL_STATIC void mat__t_scalar_(Mat *out, const Mat *m);

MAT_INTERNAL_STATIC void mat__t_neon_(Mat *out, const Mat *m) {
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

MAT_INTERNAL_STATIC void mat__t_scalar_(Mat *out, const Mat *m) {
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

MAT_INTERNAL_STATIC void mat__t_dispatch_(Mat *out, const Mat *m) {
#if defined(MAT_HAS_ARM_NEON)
  mat__t_neon_(out, m);
#elif defined(MAT_HAS_AVX2)
  mat_t_avx2_(out, m);  // Future
#else
  mat__t_scalar_(out, m);
#endif
}

MATDEF void mat_t(Mat *out, const Mat *m) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(m);
  MAT_ASSERT(out->rows == m->cols && out->cols == m->rows);

  mat__t_dispatch_(out, m);
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
MAT_INTERNAL_STATIC mat_elem_t mat__sum_neon_(const Mat *a) {
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

MAT_INTERNAL_STATIC mat_elem_t mat__sum_scalar_(const Mat *a) {
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

MAT_INTERNAL_STATIC mat_elem_t mat__sum_dispatch_(const Mat *a) {
#if defined(MAT_HAS_ARM_NEON)
  return mat__sum_neon_(a);
#elif defined(MAT_HAS_AVX2)
  return mat_sum_avx2_(a);  // Future
#else
  return mat__sum_scalar_(a);
#endif
}

MATDEF mat_elem_t mat_sum(const Mat *a) {
  MAT_ASSERT_MAT(a);

  return mat__sum_dispatch_(a);
}

MATDEF mat_elem_t mat_mean(const Mat *a) {
  MAT_ASSERT_MAT(a);
  return mat_sum(a) / (mat_elem_t)(a->rows * a->cols);
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC mat_elem_t mat__min_neon_(const Mat *a) {
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

MAT_INTERNAL_STATIC mat_elem_t mat__min_scalar_(const Mat *a) {
  size_t len = a->rows * a->cols;
  mat_elem_t *pa = a->data;

  mat_elem_t min_val = pa[0];

  for (size_t i = 1; i < len; i++) {
    if (pa[i] < min_val)
      min_val = pa[i];
  }

  return min_val;
}

MAT_INTERNAL_STATIC mat_elem_t mat__min_dispatch_(const Mat *a) {
#if defined(MAT_HAS_ARM_NEON)
  return mat__min_neon_(a);
#elif defined(MAT_HAS_AVX2)
  return mat_min_avx2_(a);  // Future
#else
  return mat__min_scalar_(a);
#endif
}

MATDEF mat_elem_t mat_min(const Mat *a) {
  MAT_ASSERT_MAT(a);

  return mat__min_dispatch_(a);
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC mat_elem_t mat__max_neon_(const Mat *a) {
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

MAT_INTERNAL_STATIC mat_elem_t mat__max_scalar_(const Mat *a) {
  size_t len = a->rows * a->cols;
  mat_elem_t *pa = a->data;

  mat_elem_t max_val = pa[0];

  for (size_t i = 1; i < len; i++) {
    if (pa[i] > max_val)
      max_val = pa[i];
  }

  return max_val;
}

MAT_INTERNAL_STATIC mat_elem_t mat__max_dispatch_(const Mat *a) {
#if defined(MAT_HAS_ARM_NEON)
  return mat__max_neon_(a);
#elif defined(MAT_HAS_AVX2)
  return mat_max_avx2_(a);  // Future
#else
  return mat__max_scalar_(a);
#endif
}

MATDEF mat_elem_t mat_max(const Mat *a) {
  MAT_ASSERT_MAT(a);

  return mat__max_dispatch_(a);
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

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC mat_elem_t mat__std_neon_(const Mat *a, mat_elem_t mean) {
  size_t len = a->rows * a->cols;
  mat_elem_t *pa = a->data;

  MAT_NEON_TYPE vmean = MAT_NEON_DUP(mean);
  MAT_NEON_TYPE vsum0 = MAT_NEON_DUP(0);
  MAT_NEON_TYPE vsum1 = MAT_NEON_DUP(0);
  MAT_NEON_TYPE vsum2 = MAT_NEON_DUP(0);
  MAT_NEON_TYPE vsum3 = MAT_NEON_DUP(0);

  size_t i = 0;
  for (; i + MAT_NEON_WIDTH * 4 <= len; i += MAT_NEON_WIDTH * 4) {
    MAT_NEON_TYPE vd0 = MAT_NEON_SUB(MAT_NEON_LOAD(&pa[i]), vmean);
    MAT_NEON_TYPE vd1 = MAT_NEON_SUB(MAT_NEON_LOAD(&pa[i + MAT_NEON_WIDTH]), vmean);
    MAT_NEON_TYPE vd2 = MAT_NEON_SUB(MAT_NEON_LOAD(&pa[i + MAT_NEON_WIDTH * 2]), vmean);
    MAT_NEON_TYPE vd3 = MAT_NEON_SUB(MAT_NEON_LOAD(&pa[i + MAT_NEON_WIDTH * 3]), vmean);

    vsum0 = MAT_NEON_FMA(vsum0, vd0, vd0);
    vsum1 = MAT_NEON_FMA(vsum1, vd1, vd1);
    vsum2 = MAT_NEON_FMA(vsum2, vd2, vd2);
    vsum3 = MAT_NEON_FMA(vsum3, vd3, vd3);
  }

  vsum0 = MAT_NEON_ADD(vsum0, vsum1);
  vsum2 = MAT_NEON_ADD(vsum2, vsum3);
  vsum0 = MAT_NEON_ADD(vsum0, vsum2);
  mat_elem_t sum_sq = MAT_NEON_ADDV(vsum0);

  for (; i < len; i++) {
    mat_elem_t diff = pa[i] - mean;
    sum_sq += diff * diff;
  }

  return sum_sq;
}
#endif

MAT_INTERNAL_STATIC mat_elem_t mat__std_scalar_(const Mat *a, mat_elem_t mean) {
  size_t len = a->rows * a->cols;
  mat_elem_t *pa = a->data;

  mat_elem_t sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;

  size_t i = 0;
  for (; i + 4 <= len; i += 4) {
    mat_elem_t d0 = pa[i] - mean;
    mat_elem_t d1 = pa[i + 1] - mean;
    mat_elem_t d2 = pa[i + 2] - mean;
    mat_elem_t d3 = pa[i + 3] - mean;
    sum0 += d0 * d0;
    sum1 += d1 * d1;
    sum2 += d2 * d2;
    sum3 += d3 * d3;
  }

  mat_elem_t sum_sq = sum0 + sum1 + sum2 + sum3;

  for (; i < len; i++) {
    mat_elem_t diff = pa[i] - mean;
    sum_sq += diff * diff;
  }

  return sum_sq;
}

MAT_INTERNAL_STATIC mat_elem_t mat__std_dispatch_(const Mat *a, mat_elem_t mean) {
#if defined(MAT_HAS_ARM_NEON)
  return mat__std_neon_(a, mean);
#elif defined(MAT_HAS_AVX2)
  return mat__std_scalar_(a, mean);  // TODO: AVX2 implementation
#else
  return mat__std_scalar_(a, mean);
#endif
}

MATDEF mat_elem_t mat_std(const Mat *a) {
  MAT_ASSERT_MAT(a);

  size_t n = a->rows * a->cols;
  mat_elem_t mean = mat_mean(a);
  mat_elem_t sum_sq = mat__std_dispatch_(a, mean);

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

MATDEF mat_elem_t mat_norm_max(const Mat *a) {
  MAT_ASSERT_MAT(a);
  return mat__amax_kernel(a->data, a->rows * a->cols);
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC mat_elem_t mat__norm_fro_neon_(const Mat *a) {
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

MAT_INTERNAL_STATIC mat_elem_t mat__norm_fro_scalar_(const Mat *a) {
  size_t len = a->rows * a->cols;
  return MAT_SQRT(mat__dot_kernel(a->data, a->data, len));
}

MAT_INTERNAL_STATIC mat_elem_t mat__norm_fro_dispatch_(const Mat *a) {
#if defined(MAT_HAS_ARM_NEON)
  return mat__norm_fro_neon_(a);
#elif defined(MAT_HAS_AVX2)
  return mat_norm_fro_avx2_(a);  // Future
#else
  return mat__norm_fro_scalar_(a);
#endif
}

MATDEF mat_elem_t mat_norm_fro(const Mat *a) {
  MAT_ASSERT_MAT(a);

  return mat__norm_fro_dispatch_(a);
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC mat_elem_t mat__norm_fro_fast_neon_(const Mat *a) {
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

MAT_INTERNAL_STATIC mat_elem_t mat__norm_fro_fast_dispatch_(const Mat *a) {
#if defined(MAT_HAS_ARM_NEON)
  return mat__norm_fro_fast_neon_(a);
#elif defined(MAT_HAS_AVX2)
  return mat_norm_fro_fast_avx2_(a);  // Future
#else
  return mat__norm_fro_scalar_(a);
#endif
}

MATDEF mat_elem_t mat_norm_fro_fast(const Mat *a) {
  MAT_ASSERT_MAT(a);

  return mat__norm_fro_fast_dispatch_(a);
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
MAT_INTERNAL_STATIC mat_elem_t mat__nnz_neon_(const Mat *a) {
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

MAT_INTERNAL_STATIC mat_elem_t mat__nnz_scalar_(const Mat *a) {
  size_t len = a->rows * a->cols;
  mat_elem_t count = 0;
  for (size_t i = 0; i < len; i++) {
    if (a->data[i] != 0)
      count++;
  }
  return count;
}

MAT_INTERNAL_STATIC mat_elem_t mat__nnz_dispatch_(const Mat *a) {
#if defined(MAT_HAS_ARM_NEON)
  return mat__nnz_neon_(a);
#elif defined(MAT_HAS_AVX2)
  return mat_nnz_avx2_(a);  // Future
#else
  return mat__nnz_scalar_(a);
#endif
}

MATDEF mat_elem_t mat_nnz(const Mat *a) {
  MAT_ASSERT_MAT(a);

  return mat__nnz_dispatch_(a);
}

/* Householder Reflections */

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
MAT_INTERNAL_STATIC void mat__qr_build_T_(mat_elem_t *T, size_t ldt,
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
mat__transpose_blocked_(mat_elem_t *Bt, const mat_elem_t *B, size_t rows,
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
mat__qr_apply_block_left_(mat_elem_t *A_data, size_t lda,
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
  mat__gemm_strided_(C, k, 1.0f, Y, ldy, MAT_TRANS, A_data, lda,
                             MAT_NO_TRANS, k, m, n, 0.0f);

  // TC = T^T * C (k x n) - use transA flag instead of explicit transpose
  mat__gemm_strided_(TC, k, 1.0f, T, ldt, MAT_TRANS, C, k, MAT_NO_TRANS,
                             k, k, n, 0.0f);

  // A = A - Y * TC (m x n)
  mat__gemm_strided_(A_data, lda, -1.0f, Y, ldy, MAT_NO_TRANS, TC, k,
                             MAT_NO_TRANS, m, k, n, 1.0f);
}

// Column-major: Apply block Householder from right: A = A * (I - Y*T*Y^T)
// Uses pre-allocated workspace
MAT_INTERNAL_STATIC void
mat__qr_apply_block_right_(mat_elem_t *A_data, size_t lda,
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
  mat__gemm_strided_(C, m, 1.0f, A_data, lda, MAT_NO_TRANS, Y, ldy,
                             MAT_NO_TRANS, m, n, k, 0.0f);

  // CT = C * T (m x k)
  mat__gemm_strided_(CT, m, 1.0f, C, m, MAT_NO_TRANS, T, ldt,
                             MAT_NO_TRANS, m, k, k, 0.0f);

  // A = A - CT * Y^T (m x n) - use transB flag instead of explicit transpose
  mat__gemm_strided_(A_data, lda, -1.0f, CT, m, MAT_NO_TRANS, Y, ldy,
                             MAT_TRANS, m, k, n, 1.0f);
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
        mat__copy_kernel(x_data, &R->data[col * m + col], vlen);

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
      mat__qr_build_T_(T, kb, Y, len, tau_block, len, kb);

      // Apply block reflector to trailing R (work directly with stride m)
      if (jb + kb < n) {
        size_t trail_cols = n - (jb + kb);
        mat_elem_t *R_src = &R->data[(jb + kb) * m + jb];

        mat__qr_apply_block_left_(R_src, m, Y, len, T, kb, len, kb,
                                          trail_cols, &ws);
      }

      // Apply block reflector to Q (in-place - Q columns are always dense)
      // Q[:, jb:m] starts at Q->data[jb*m], all m rows accessed
      {
        mat_elem_t *Q_trail = &Q->data[jb * m];
        mat__qr_apply_block_right_(Q_trail, m, Y, len, T, kb, m, len,
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
      mat_elem_t w = tau * mat__dot_kernel(v_data, Rk, len);
      mat__axpy_kernel(Rk, -w, v_data, len);
    }

    // Apply H to Q[:, j:m] from right (column-major)
    // Q = Q * H = Q - tau * (Q * v) * v^T
    // Step 1: u = Q[:, j:m] * v (compute u column-by-column for contiguous access)
    memset(u, 0, m * sizeof(mat_elem_t));

    // u = Q[:, j:m] * v, computed column-by-column
    for (size_t jj = 0; jj < len; jj++) {
      mat__axpy_kernel(u, v_data[jj], &Q->data[(j + jj) * m], m);
    }

    // Step 2: Q[:, j:m] -= tau * u * v^T, applied column by column
    for (size_t jj = 0; jj < len; jj++) {
      mat__axpy_kernel(&Q->data[(j + jj) * m], -tau * v_data[jj], u, m);
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
      mat__qr_build_T_(T, kb, Y, len, tau_block, len, kb);

      // Apply block reflector to trailing R only (skip Q!)
      // Work directly on R with stride m (no copy needed)
      if (jb + kb < n) {
        size_t trail_cols = n - (jb + kb);
        mat_elem_t *R_src = &R->data[(jb + kb) * m + jb];

        mat_elem_t *C = W_kn;
        mat_elem_t *TC = W_kn2;

        // C = Y^T * R_src (kb x trail_cols), R_src has stride m
        mat__gemm_strided_(C, kb, 1.0f, Y, len, MAT_TRANS, R_src, m,
                                   MAT_NO_TRANS, kb, len, trail_cols, 0.0f);
        // TC = T^T * C (kb x trail_cols)
        mat__gemm_strided_(TC, kb, 1.0f, T, kb, MAT_TRANS, C, kb,
                                   MAT_NO_TRANS, kb, kb, trail_cols, 0.0f);
        // R_src -= Y * TC (len x trail_cols), R_src has stride m
        mat__gemm_strided_(R_src, m, -1.0f, Y, len, MAT_NO_TRANS, TC,
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
      mat_elem_t w = tau * mat__dot_kernel(v_data, Rk, len);
      mat__axpy_kernel(Rk, -w, v_data, len);
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
MAT_INTERNAL_STATIC int mat__plu_blocked_(Mat *M, Perm *p) {
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
      size_t pivot_row = k + mat__iamax(&col_k[k], n - k, NULL);
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

      // Scale L column (contiguous)
      mat__scal_kernel(&col_k[k + 1], pivot_inv, n - (k + 1));

      // Update panel columns k+1:k_end for all rows below k
      for (size_t j = k + 1; j < k_end; j++) {
        mat_elem_t *col_j = &data[j * n];
        mat__axpy_kernel(&col_j[k + 1], -col_j[k], &col_k[k + 1], n - (k + 1));
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
          mat__axpy_kernel(&col_j[k + 1], -col_j[k], &col_k[k + 1], k_end - (k + 1));
        }
      }

      // 3. GEMM: A22 -= L21 * U12
      size_t trail_m = n - k_end;
      size_t trail_n = n - k_end;

      mat__gemm_strided_(
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
MAT_INTERNAL_STATIC int mat__lu_scalar_(Mat *M, Perm *p, Perm *q) {
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
// NEON-optimized full pivoting LU decomposition
// In column-major: column j is contiguous at &data[j * n]
MAT_INTERNAL_STATIC int mat__lu_neon_(Mat *M, Perm *p, Perm *q) {
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
      mat_elem_t col_max;
      size_t local_idx = mat__iamax(&col[k], n - k, &col_max);

      if (col_max > max_val) {
        max_val = col_max;
        pivot_col = j;
        pivot_row = k + local_idx;
      }
    }

    // Swap columns k and pivot_col (columns are contiguous!)
    if (pivot_col != k) {
      mat__swap_kernel(&data[k * n], &data[pivot_col * n], n);

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
    mat__scal_kernel(&col_k[k + 1], pivot_inv, n - k - 1);

    // Elimination: column-oriented rank-1 update
    // For each column j > k: col_j[k+1:n] -= col_j[k] * col_k[k+1:n]
    for (size_t j = k + 1; j < n; j++) {
      mat_elem_t *col_j = &data[j * n];
      mat__axpy_kernel(&col_j[k + 1], -col_j[k], &col_k[k + 1], n - k - 1);
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
  int swap_count = mat__plu_blocked_(M, p);

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

    int swap_count;
  mat_elem_t *data = M->data;
#if defined(MAT_HAS_ARM_NEON)
  // Column-major LU decomposition (NEON optimized)
  swap_count = mat__lu_neon_(M, p, q);
#elif defined(MAT_HAS_AVX2)
  // Column-major LU decomposition (AVX2 - future)
  swap_count = mat_lu_avx2_(M, p, q);
#else
  // Scalar fallback - column-major LU decomposition
  swap_count = 0;
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

/* Triangular Solvers */

// --- mat_solve_tril: Solve Lx = b, L lower triangular (non-unit diagonal) ---

MAT_INTERNAL_STATIC void mat__solve_tril_scalar_(Vec *x, const Mat *L,
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
MAT_INTERNAL_STATIC void mat__solve_tril_neon_(Vec *x, const Mat *L,
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

MAT_INTERNAL_STATIC void mat__solve_tril_dispatch_(Vec *x, const Mat *L,
                                                   const Vec *b) {
#if defined(MAT_HAS_ARM_NEON)
  mat__solve_tril_neon_(x, L, b);
#elif defined(MAT_HAS_AVX2)
  mat_solve_tril_avx2_(x, L, b);  // Future
#else
  mat__solve_tril_scalar_(x, L, b);
#endif
}

MATDEF void mat_solve_tril(Vec *x, const Mat *L, const Vec *b) {
  MAT_ASSERT_MAT(L);
  MAT_ASSERT_MAT(x);
  MAT_ASSERT_MAT(b);
  MAT_ASSERT_SQUARE(L);
  MAT_ASSERT(b->rows == L->rows);
  MAT_ASSERT(x->rows == L->rows);
  mat__solve_tril_dispatch_(x, L, b);
}

// --- mat_solve_tril_unit: Solve Lx = b, L unit lower triangular (1s on diag) -

MAT_INTERNAL_STATIC void mat__solve_tril_unit_scalar_(Vec *x, const Mat *L,
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
MAT_INTERNAL_STATIC void mat__solve_tril_unit_neon_(Vec *x, const Mat *L,
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

MAT_INTERNAL_STATIC void mat__solve_tril_unit_dispatch_(Vec *x, const Mat *L,
                                                        const Vec *b) {
#if defined(MAT_HAS_ARM_NEON)
  mat__solve_tril_unit_neon_(x, L, b);
#elif defined(MAT_HAS_AVX2)
  mat_solve_tril_unit_avx2_(x, L, b);  // Future
#else
  mat__solve_tril_unit_scalar_(x, L, b);
#endif
}

MATDEF void mat_solve_tril_unit(Vec *x, const Mat *L, const Vec *b) {
  MAT_ASSERT_MAT(L);
  MAT_ASSERT_MAT(x);
  MAT_ASSERT_MAT(b);
  MAT_ASSERT_SQUARE(L);
  MAT_ASSERT(b->rows == L->rows);
  MAT_ASSERT(x->rows == L->rows);
  mat__solve_tril_unit_dispatch_(x, L, b);
}

// --- mat_solve_triu: Solve Ux = b, U upper triangular ---
// Row-oriented backward substitution with NEON optimization.
// For row-major U, accessing U[i, i+1:n] is contiguous.

#ifdef MAT_HAS_ARM_NEON
// Forward declaration for fallback
MAT_INTERNAL_STATIC void mat__solve_triu_scalar_(Vec *x, const Mat *U,
                                                 const Vec *b);

MAT_INTERNAL_STATIC void mat__solve_triu_neon_(Vec *x, const Mat *U,
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

MAT_INTERNAL_STATIC void mat__solve_triu_scalar_(Vec *x, const Mat *U,
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

MAT_INTERNAL_STATIC void mat__solve_triu_dispatch_(Vec *x, const Mat *U,
                                                   const Vec *b) {
#if defined(MAT_HAS_ARM_NEON)
  mat__solve_triu_neon_(x, U, b);
#elif defined(MAT_HAS_AVX2)
  mat_solve_triu_avx2_(x, U, b);  // Future
#else
  mat__solve_triu_scalar_(x, U, b);
#endif
}

MATDEF void mat_solve_triu(Vec *x, const Mat *U, const Vec *b) {
  MAT_ASSERT_MAT(U);
  MAT_ASSERT_MAT(x);
  MAT_ASSERT_MAT(b);
  MAT_ASSERT_SQUARE(U);
  MAT_ASSERT(b->rows == U->rows);
  MAT_ASSERT(x->rows == U->rows);
  mat__solve_triu_dispatch_(x, U, b);
}

// --- mat_solve_trilt: Solve L^T x = b, L lower triangular ---
// Uses column-oriented algorithm for contiguous memory access.

#ifdef MAT_HAS_ARM_NEON
// Forward declaration for fallback
MAT_INTERNAL_STATIC void mat__solve_trilt_scalar_(Vec *x, const Mat *L,
                                                  const Vec *b);

MAT_INTERNAL_STATIC void mat__solve_trilt_neon_(Vec *x, const Mat *L,
                                                const Vec *b) {
  // Solve L^T * x = b using dot product approach
  // L^T[j,k] = L[k,j], so we need dot(L[j+1:n, j], x[j+1:n]) which is contiguous
  size_t n = L->rows;
  mat_elem_t *x_data = x->data;
  const mat_elem_t *b_data = b->data;
  const mat_elem_t *L_data = L->data;

  for (size_t j = n; j-- > 0;) {
    const mat_elem_t *Lj = &L_data[j * n];
    mat_elem_t dot = mat__dot_kernel(&Lj[j + 1], &x_data[j + 1], n - j - 1);
    x_data[j] = (b_data[j] - dot) / Lj[j];
  }
}
#endif // MAT_HAS_ARM_NEON

MAT_INTERNAL_STATIC void mat__solve_trilt_scalar_(Vec *x, const Mat *L,
                                                  const Vec *b) {
  // Solve L^T * x = b using dot product approach
  // L^T[j,k] = L[k,j], so we need dot(L[j+1:n, j], x[j+1:n]) which is contiguous
  size_t n = L->rows;
  mat_elem_t *x_data = x->data;
  const mat_elem_t *b_data = b->data;
  const mat_elem_t *L_data = L->data;

  for (size_t j = n; j-- > 0;) {
    const mat_elem_t *Lj = &L_data[j * n];  // Column j is contiguous
    mat_elem_t dot = 0;
    for (size_t k = j + 1; k < n; k++) {
      dot += Lj[k] * x_data[k];
    }
    x_data[j] = (b_data[j] - dot) / Lj[j];
  }
}

MAT_INTERNAL_STATIC void mat__solve_trilt_dispatch_(Vec *x, const Mat *L,
                                                    const Vec *b) {
#if defined(MAT_HAS_ARM_NEON)
  mat__solve_trilt_neon_(x, L, b);
#elif defined(MAT_HAS_AVX2)
  mat_solve_trilt_avx2_(x, L, b);  // Future
#else
  mat__solve_trilt_scalar_(x, L, b);
#endif
}

MATDEF void mat_solve_trilt(Vec *x, const Mat *L, const Vec *b) {
  MAT_ASSERT_MAT(L);
  MAT_ASSERT_MAT(x);
  MAT_ASSERT_MAT(b);
  MAT_ASSERT_SQUARE(L);
  MAT_ASSERT(b->rows == L->rows);
  MAT_ASSERT(x->rows == L->rows);

  // L^T solve uses dot product approach: contiguous access to L columns
  // (AXPY approach would need strided access to L rows, which is slower)
  mat__solve_trilt_dispatch_(x, L, b);
}

/* Linear Solvers */

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

/* Cholesky Decomposition */

#define MAT_CHOL_BLOCK_SIZE 32

// Lower-triangular GEMM: C[lower] += alpha * A * B^T[lower]
// Only computes and updates the lower triangle of C
// Uses GEMM-style 4x4 micro-kernel with packed B for cache efficiency
#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void
mat__gemm_lower_strided_neon_(mat_elem_t *C, size_t ldc, mat_elem_t alpha,
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



/* SYRK */

// Column-major SYRK lower triangle - scalar implementation
MAT_INTERNAL_STATIC void mat__syrk_lower_scalar_(
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
MAT_INTERNAL_STATIC void mat__syrk_lower_unblocked_(
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
MAT_INTERNAL_STATIC void mat__syrk_lower_blocked_(
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
#define MAT_SYRK_BLOCK_THRESHOLD 96
MAT_INTERNAL_STATIC void mat__syrk_lower_(
    mat_elem_t *C, size_t ldc, mat_elem_t alpha, const mat_elem_t *A,
    size_t lda, size_t m, size_t k, mat_elem_t beta) {
#ifdef MAT_HAS_ARM_NEON
  if (m >= MAT_SYRK_BLOCK_THRESHOLD) {
    mat__syrk_lower_blocked_(C, ldc, alpha, A, lda, m, k, beta);
  } else {
    mat__syrk_lower_unblocked_(C, ldc, alpha, A, lda, m, k, beta);
  }
#else
  mat__syrk_lower_scalar_(C, ldc, alpha, A, lda, m, k, beta);
#endif
}
#undef MAT_SYRK_BLOCK_THRESHOLD

// Transpose: At[i,p] = A[p,i] - Scalar implementation
MAT_INTERNAL_STATIC void
mat__transpose_block_scalar_(mat_elem_t *At, size_t ldat, const mat_elem_t *A,
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
MAT_INTERNAL_STATIC void mat__transpose_block_neon_(mat_elem_t *At, size_t ldat,
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
MAT_INTERNAL_STATIC void mat__transpose_block_(mat_elem_t *At, size_t ldat,
                                              const mat_elem_t *A, size_t lda,
                                              size_t rows, size_t cols) {
#if defined(MAT_HAS_ARM_NEON)
  mat__transpose_block_neon_(At, ldat, A, lda, rows, cols);
#elif defined(MAT_HAS_AVX2)
  mat__transpose_block_avx2_(At, ldat, A, lda, rows, cols);  // Future
#else
  mat__transpose_block_scalar_(At, ldat, A, lda, rows, cols);
#endif
}

// Symmetric rank-k update for lower triangle (transposed): C = alpha * A^T * A
// + beta * C A is k x n, C is n x n Strategy: fast NEON transpose then
// optimized SYRK
MAT_INTERNAL_STATIC void mat__syrk_t_lower_(mat_elem_t *C, size_t ldc,
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
  mat__transpose_block_(At, k, A, lda, n, k);

  // Use optimized lower SYRK
  mat__syrk_lower_(C, ldc, alpha, At, k, n, k, beta);

#ifndef MAT_NO_SCRATCH
  mat_scratch_reset_();
#else
  mat_scratch_free_(At);
#endif
}

// Symmetric rank-k update for upper triangle (transposed): C = alpha * A^T * A
// + beta * C A is k x n, C is n x n
MAT_INTERNAL_STATIC void mat__syrk_t_upper_(mat_elem_t *C, size_t ldc,
                                           mat_elem_t alpha,
                                           const mat_elem_t *A, size_t lda,
                                           size_t n, size_t k,
                                           mat_elem_t beta) {
  // Compute lower triangle then copy to upper
  // (Upper-specific kernel was removed with row-major code)
  mat__syrk_t_lower_(C, ldc, alpha, A, lda, n, k, beta);

  // Copy lower to upper (column-major: C[i,j] = C[j*ldc + i])
  for (size_t j = 0; j < n; j++) {
    for (size_t i = j + 1; i < n; i++) {
      C[i * ldc + j] = C[j * ldc + i];
    }
  }
}

// Column-major SYRK: C = alpha * A * A^T + beta * C
#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void mat__syrk_neon_(Mat *C, const Mat *A,
                                        mat_elem_t alpha,
                                        mat_elem_t beta, char uplo) {
  size_t n = C->rows;
  size_t k = A->cols;

  // Use column-major SYRK directly (only lower triangle currently implemented)
  if (uplo == 'L' || uplo == 'l') {
    mat__syrk_lower_(C->data, n, alpha, A->data, n, n, k, beta);
  } else {
    // For upper triangle, compute lower triangle and copy
    // (Since upper-specific kernel was removed with row-major code)
    mat__syrk_lower_(C->data, n, alpha, A->data, n, n, k, beta);
    // Copy lower to upper
    for (size_t i = 0; i < n; i++) {
      for (size_t j = i + 1; j < n; j++) {
        C->data[i * n + j] = C->data[j * n + i];
      }
    }
  }
}
#endif

// Generic scalar SYRK using MAT_AT/MAT_SET (works with any storage order)
MAT_INTERNAL_STATIC void mat__syrk_generic_(Mat *C, const Mat *A,
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

MAT_INTERNAL_STATIC void mat__syrk_dispatch_(Mat *C, const Mat *A,
                                             mat_elem_t alpha, mat_elem_t beta,
                                             char uplo) {
#if defined(MAT_HAS_ARM_NEON)
  mat__syrk_neon_(C, A, alpha, beta, uplo);
#elif defined(MAT_HAS_AVX2)
  mat_syrk_avx2_(C, A, alpha, beta, uplo);  // Future
#else
  mat__syrk_generic_(C, A, alpha, beta, uplo);
#endif
}

MATDEF void mat_syrk(Mat *C, const Mat *A, mat_elem_t alpha, mat_elem_t beta,
                     char uplo) {
  MAT_ASSERT_MAT(C);
  MAT_ASSERT_MAT(A);
  MAT_ASSERT(C->rows == C->cols);
  MAT_ASSERT(C->rows == A->rows);

  mat__syrk_dispatch_(C, A, alpha, beta, uplo);
}

// Column-major SYRK_T: C = alpha * A^T * A + beta * C
#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void mat__syrk_t_neon_(Mat *C, const Mat *A,
                                          mat_elem_t alpha,
                                          mat_elem_t beta, char uplo) {
  size_t n = C->rows;  // = A->cols
  size_t k = A->rows;  // inner dimension

  if (uplo == 'L' || uplo == 'l') {
    mat__syrk_t_lower_(C->data, n, alpha, A->data, k, n, k, beta);
  } else {
    mat__syrk_t_upper_(C->data, n, alpha, A->data, k, n, k, beta);
  }
}
#endif

// Generic scalar SYRK_T using MAT_AT/MAT_SET (works with any storage order)
// Computes C = alpha * A^T * A + beta * C
MAT_INTERNAL_STATIC void mat__syrk_t_generic_(Mat *C, const Mat *A,
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

MAT_INTERNAL_STATIC void mat__syrk_t_dispatch_(Mat *C, const Mat *A,
                                               mat_elem_t alpha, mat_elem_t beta,
                                               char uplo) {
#if defined(MAT_HAS_ARM_NEON)
  mat__syrk_t_neon_(C, A, alpha, beta, uplo);
#elif defined(MAT_HAS_AVX2)
  mat_syrk_t_avx2_(C, A, alpha, beta, uplo);  // Future
#else
  mat__syrk_t_generic_(C, A, alpha, beta, uplo);
#endif
}

MATDEF void mat_syrk_t(Mat *C, const Mat *A, mat_elem_t alpha, mat_elem_t beta,
                       char uplo) {
  MAT_ASSERT_MAT(C);
  MAT_ASSERT_MAT(A);
  MAT_ASSERT(C->rows == C->cols);
  MAT_ASSERT(C->rows == A->cols);

  mat__syrk_t_dispatch_(C, A, alpha, beta, uplo);
}

/* SYR2K */

// Generic scalar SYR2K: C = alpha*A*B' + alpha*B*A' + beta*C
// A is n x k, B is n x k, C is n x n symmetric
MAT_INTERNAL_STATIC void mat__syr2k_generic_(Mat *C, const Mat *A, const Mat *B,
                                             mat_elem_t alpha, mat_elem_t beta,
                                             char uplo) {
  size_t n = C->rows;
  size_t k = A->cols;

  if (uplo == 'L' || uplo == 'l') {
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j <= i; j++) {
        mat_elem_t sum = 0;
        for (size_t kk = 0; kk < k; kk++) {
          sum += MAT_AT(A, i, kk) * MAT_AT(B, j, kk) +
                 MAT_AT(B, i, kk) * MAT_AT(A, j, kk);
        }
        MAT_SET(C, i, j, beta * MAT_AT(C, i, j) + alpha * sum);
      }
    }
  } else {
    for (size_t i = 0; i < n; i++) {
      for (size_t j = i; j < n; j++) {
        mat_elem_t sum = 0;
        for (size_t kk = 0; kk < k; kk++) {
          sum += MAT_AT(A, i, kk) * MAT_AT(B, j, kk) +
                 MAT_AT(B, i, kk) * MAT_AT(A, j, kk);
        }
        MAT_SET(C, i, j, beta * MAT_AT(C, i, j) + alpha * sum);
      }
    }
  }
}

#ifdef MAT_HAS_ARM_NEON
// NEON-optimized SYR2K for lower triangle, column-major
// C = alpha*A*B' + alpha*B*A' + beta*C
// A is n x k (column-major: A[i,kk] = A_data[kk*n + i])
// B is n x k (column-major: B[i,kk] = B_data[kk*n + i])
MAT_INTERNAL_STATIC void mat__syr2k_lower_neon_(mat_elem_t *C, size_t ldc,
                                                const mat_elem_t *A, size_t lda,
                                                const mat_elem_t *B, size_t ldb,
                                                size_t n, size_t k,
                                                mat_elem_t alpha, mat_elem_t beta) {
  // For each column j of C (lower triangle: rows j to n-1)
  for (size_t j = 0; j < n; j++) {
    mat_elem_t *Cj = &C[j * ldc + j];  // C[j:n, j]
    size_t len = n - j;

    // Compute C[j:n, j] = beta*C[j:n, j] + alpha*(A[j:n,:]*B[j,:]' + B[j:n,:]*A[j,:]')
    // This is: sum over kk of alpha*(A[i,kk]*B[j,kk] + B[i,kk]*A[j,kk])

    // Scale C by beta first
    if (beta == 0) {
      memset(Cj, 0, len * sizeof(mat_elem_t));
    } else if (beta != 1) {
      mat__scal_kernel(Cj, beta, len);
    }

    // Accumulate rank-2 updates
    for (size_t kk = 0; kk < k; kk++) {
      const mat_elem_t *Ak = &A[kk * lda + j];  // A[j:n, kk]
      const mat_elem_t *Bk = &B[kk * ldb + j];  // B[j:n, kk]
      mat_elem_t Bjk = B[kk * ldb + j];  // B[j, kk] (scalar)
      mat_elem_t Ajk = A[kk * lda + j];  // A[j, kk] (scalar)

      MAT_NEON_TYPE vBjk = MAT_NEON_DUP(alpha * Bjk);
      MAT_NEON_TYPE vAjk = MAT_NEON_DUP(alpha * Ajk);

      size_t i = 0;
      for (; i + MAT_NEON_WIDTH * 4 <= len; i += MAT_NEON_WIDTH * 4) {
        MAT_NEON_TYPE vc0 = MAT_NEON_LOAD(&Cj[i]);
        MAT_NEON_TYPE vc1 = MAT_NEON_LOAD(&Cj[i + MAT_NEON_WIDTH]);
        MAT_NEON_TYPE vc2 = MAT_NEON_LOAD(&Cj[i + MAT_NEON_WIDTH * 2]);
        MAT_NEON_TYPE vc3 = MAT_NEON_LOAD(&Cj[i + MAT_NEON_WIDTH * 3]);

        MAT_NEON_TYPE va0 = MAT_NEON_LOAD(&Ak[i]);
        MAT_NEON_TYPE va1 = MAT_NEON_LOAD(&Ak[i + MAT_NEON_WIDTH]);
        MAT_NEON_TYPE va2 = MAT_NEON_LOAD(&Ak[i + MAT_NEON_WIDTH * 2]);
        MAT_NEON_TYPE va3 = MAT_NEON_LOAD(&Ak[i + MAT_NEON_WIDTH * 3]);

        MAT_NEON_TYPE vb0 = MAT_NEON_LOAD(&Bk[i]);
        MAT_NEON_TYPE vb1 = MAT_NEON_LOAD(&Bk[i + MAT_NEON_WIDTH]);
        MAT_NEON_TYPE vb2 = MAT_NEON_LOAD(&Bk[i + MAT_NEON_WIDTH * 2]);
        MAT_NEON_TYPE vb3 = MAT_NEON_LOAD(&Bk[i + MAT_NEON_WIDTH * 3]);

        // C += alpha * (A * B[j,kk] + B * A[j,kk])
        vc0 = MAT_NEON_FMA(MAT_NEON_FMA(vc0, va0, vBjk), vb0, vAjk);
        vc1 = MAT_NEON_FMA(MAT_NEON_FMA(vc1, va1, vBjk), vb1, vAjk);
        vc2 = MAT_NEON_FMA(MAT_NEON_FMA(vc2, va2, vBjk), vb2, vAjk);
        vc3 = MAT_NEON_FMA(MAT_NEON_FMA(vc3, va3, vBjk), vb3, vAjk);

        MAT_NEON_STORE(&Cj[i], vc0);
        MAT_NEON_STORE(&Cj[i + MAT_NEON_WIDTH], vc1);
        MAT_NEON_STORE(&Cj[i + MAT_NEON_WIDTH * 2], vc2);
        MAT_NEON_STORE(&Cj[i + MAT_NEON_WIDTH * 3], vc3);
      }

      for (; i + MAT_NEON_WIDTH <= len; i += MAT_NEON_WIDTH) {
        MAT_NEON_TYPE vc = MAT_NEON_LOAD(&Cj[i]);
        MAT_NEON_TYPE va = MAT_NEON_LOAD(&Ak[i]);
        MAT_NEON_TYPE vb = MAT_NEON_LOAD(&Bk[i]);
        vc = MAT_NEON_FMA(MAT_NEON_FMA(vc, va, vBjk), vb, vAjk);
        MAT_NEON_STORE(&Cj[i], vc);
      }

      mat_elem_t aBjk = alpha * Bjk;
      mat_elem_t aAjk = alpha * Ajk;
      for (; i < len; i++) {
        Cj[i] += aBjk * Ak[i] + aAjk * Bk[i];
      }
    }
  }
}

MAT_INTERNAL_STATIC void mat__syr2k_neon_(Mat *C, const Mat *A, const Mat *B,
                                           mat_elem_t alpha, mat_elem_t beta,
                                           char uplo) {
  size_t n = C->rows;
  size_t k = A->cols;

  if (uplo == 'L' || uplo == 'l') {
    mat__syr2k_lower_neon_(C->data, n, A->data, n, B->data, n, n, k, alpha, beta);
  } else {
    // Upper triangle: transpose the problem or use generic
    // For now, use generic for upper triangle
    mat__syr2k_generic_(C, A, B, alpha, beta, uplo);
  }
}
#endif

MAT_INTERNAL_STATIC void mat__syr2k_dispatch_(Mat *C, const Mat *A, const Mat *B,
                                               mat_elem_t alpha, mat_elem_t beta,
                                               char uplo) {
#ifdef MAT_HAS_ARM_NEON
  mat__syr2k_neon_(C, A, B, alpha, beta, uplo);
#else
  mat__syr2k_generic_(C, A, B, alpha, beta, uplo);
#endif
}

MATDEF void mat_syr2k(Mat *C, const Mat *A, const Mat *B, mat_elem_t alpha,
                      mat_elem_t beta, char uplo) {
  MAT_ASSERT_MAT(C);
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(B);
  MAT_ASSERT(C->rows == C->cols);
  MAT_ASSERT(C->rows == A->rows);
  MAT_ASSERT(C->rows == B->rows);
  MAT_ASSERT(A->cols == B->cols);

  mat__syr2k_dispatch_(C, A, B, alpha, beta, uplo);
}

// Column-major unblocked Cholesky using dot products for better cache locality
// In column-major: L[0:j, i] is contiguous at M[i*ldm], so dots are efficient
MAT_INTERNAL_STATIC int mat__chol_unblocked_(mat_elem_t *M, size_t n,
                                              size_t ldm) {
  for (size_t j = 0; j < n; j++) {
    // Diagonal: L[j,j] = sqrt(A[j,j] - dot(L[0:j, j], L[0:j, j]))
    // L[0:j, j] is at M[j*ldm + 0:j] which is contiguous
    mat_elem_t *Lj = &M[j * ldm];  // Column j, elements 0:n
    mat_elem_t diag_sum = mat__dot_kernel(Lj, Lj, j);
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
      mat_elem_t dot = mat__dot_kernel(Li, Lj, j);
      Lj[i] = (Lj[i] - dot) * ljj_inv;
    }
#else
    for (size_t i = j + 1; i < n; i++) {
      mat_elem_t *Li = &M[i * ldm];
      mat_elem_t dot = mat__dot_kernel(Li, Lj, j);
      Lj[i] = (Lj[i] - dot) * ljj_inv;
    }
#endif
  }
  return 0;
}

// Unblocked AXPY-based Cholesky factorization
MAT_INTERNAL_STATIC int mat__chol_unblocked_axpy_(mat_elem_t *M, size_t n,
                                                   size_t ldm) {
  for (size_t j = 0; j < n; j++) {
    mat_elem_t *colj = &M[j * ldm];

    // Update column j using previous columns: colj[j:n] -= ljk * colk[j:n]
    for (size_t k = 0; k < j; k++) {
      mat_elem_t ljk = M[k * ldm + j];
      mat__axpy_kernel(&colj[j], -ljk, &M[k * ldm + j], n - j);
    }

    if (colj[j] <= 0) {
      return -1;
    }
    mat_elem_t ljj = MAT_SQRT(colj[j]);
    colj[j] = ljj;

    // Scale: colj[j+1:n] *= 1/ljj
    mat__scal_kernel(&colj[j + 1], 1.0f / ljj, n - j - 1);
  }
  return 0;
}

// Column-major blocked Cholesky (right-looking, uses mat_syrk for trailing update)
// Structure: panel factorization -> TRSM -> SYRK
#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC int mat__chol_blocked_(mat_elem_t *M, size_t n,
                                            size_t ldm) {
  for (size_t kb = 0; kb < n; kb += MAT_CHOL_BLOCK_SIZE) {
    size_t k_end =
        (kb + MAT_CHOL_BLOCK_SIZE < n) ? kb + MAT_CHOL_BLOCK_SIZE : n;
    size_t block_k = k_end - kb;

    // 1. Panel factorization: unblocked Cholesky on diagonal block L11
    int result = mat__chol_unblocked_axpy_(&M[kb * ldm + kb], block_k, ldm);
    if (result != 0) {
      return -1;
    }

    // 2. TRSM: Solve L21 from L11 * L21^T = A21^T
    if (k_end < n) {
      size_t trail_m = n - k_end;

      for (size_t j = 0; j < block_k; j++) {
        mat_elem_t *L11_col_j = &M[(kb + j) * ldm + kb];
        mat_elem_t *L21_col_j = &M[(kb + j) * ldm + k_end];
        mat_elem_t ljj_inv = 1.0f / L11_col_j[j];

        // Update using previous columns: L21[:,j] -= L11[j,kk] * L21[:,kk]
        for (size_t kk = 0; kk < j; kk++) {
          mat_elem_t L11_jk = M[(kb + kk) * ldm + (kb + j)];
          mat__axpy_kernel(L21_col_j, -L11_jk, &M[(kb + kk) * ldm + k_end], trail_m);
        }

        // Scale by 1/L11[j,j]
        mat__scal_kernel(L21_col_j, ljj_inv, trail_m);
      }

      // 3. SYRK: A22 -= L21 * L21^T
      mat__syrk_lower_(&M[k_end * ldm + k_end], ldm, -1,
                      &M[kb * ldm + k_end], ldm,
                      trail_m, block_k, 1);
    }
  }
  return 0;
}
#endif

// Scalar Cholesky factorization using MAT_AT/MAT_SET (works with any storage)
MAT_INTERNAL_STATIC int mat__chol_scalar_generic_(Mat *L) {
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
#ifdef MAT_HAS_ARM_NEON
  mat_elem_t *Ldata = L->data;
  if (n < MAT_CHOL_BLOCK_SIZE) {
    return mat__chol_unblocked_axpy_(Ldata, n, n);
  } else {
    return mat__chol_blocked_(Ldata, n, n);
  }
#else
  return mat__chol_scalar_generic_(L);
#endif
}

// Blocked TRSM: solve L*X = B where L is unit lower triangular (column-major)
// X overwrites B in place. L is n×n, B is n×nrhs
MAT_INTERNAL_STATIC void mat__trsm_lower_unit_(mat_elem_t *B, size_t ldb,
                                                const mat_elem_t *L, size_t ldl,
                                                size_t n, size_t nrhs) {
  const size_t NB = 64;  // Block size

  for (size_t kb = 0; kb < n; kb += NB) {
    size_t k_end = (kb + NB < n) ? kb + NB : n;
    size_t block_k = k_end - kb;

    // Solve diagonal block: L11 * X1 = B1 (unit lower triangular)
    for (size_t j = 0; j < nrhs; j++) {
      mat_elem_t *Bj = &B[j * ldb];
      for (size_t i = kb; i < k_end; i++) {
        mat_elem_t sum = Bj[i];
        for (size_t kk = kb; kk < i; kk++) {
          sum -= L[kk * ldl + i] * Bj[kk];
        }
        Bj[i] = sum;
      }
    }

    // Update trailing rows: B2 -= L21 * X1
    if (k_end < n) {
      size_t trail_m = n - k_end;
      mat__gemm_strided_(&B[k_end], ldb, -1.0f,
                        &L[kb * ldl + k_end], ldl, MAT_NO_TRANS,
                        &B[kb], ldb, MAT_NO_TRANS,
                        trail_m, block_k, nrhs, 1.0f);
    }
  }
}

// Blocked TRSM: solve U*X = B where U is upper triangular (column-major)
// X overwrites B in place. U is n×n, B is n×nrhs
MAT_INTERNAL_STATIC void mat__trsm_upper_(mat_elem_t *B, size_t ldb,
                                           const mat_elem_t *U, size_t ldu,
                                           size_t n, size_t nrhs) {
  const size_t NB = 64;  // Block size

  for (size_t kb = n; kb > 0;) {
    size_t k_start = (kb >= NB) ? kb - NB : 0;
    size_t block_k = kb - k_start;
    kb = k_start;

    // Solve diagonal block: U11 * X1 = B1 (upper triangular)
    for (size_t j = 0; j < nrhs; j++) {
      mat_elem_t *Bj = &B[j * ldb];
      for (size_t ii = block_k; ii > 0; ii--) {
        size_t i = k_start + ii - 1;
        mat_elem_t sum = Bj[i];
        for (size_t kk = i + 1; kk < k_start + block_k; kk++) {
          sum -= U[kk * ldu + i] * Bj[kk];
        }
        Bj[i] = sum / U[i * ldu + i];
      }
    }

    // Update leading rows: B0 -= U01 * X1
    if (k_start > 0) {
      mat__gemm_strided_(B, ldb, -1.0f,
                        &U[k_start * ldu], ldu, MAT_NO_TRANS,
                        &B[k_start], ldb, MAT_NO_TRANS,
                        k_start, block_k, nrhs, 1.0f);
    }
  }
}

MATDEF void mat_inv(Mat *out, const Mat *A) {
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_SQUARE(A);
  size_t n = A->rows;
  MAT_ASSERT(out->rows == n && out->cols == n);

  // Compute in-place PLU decomposition: PA = LU (L and U packed in M)
  Mat *M = mat_rdeep_copy(A);
  Perm *p = mat_perm(n);
  mat_perm_identity(p);
  mat__plu_blocked_(M, p);

  const mat_elem_t *LU = M->data;

  // Initialize out to permuted identity: out[:,i] = e_{p[i]}
  memset(out->data, 0, n * n * sizeof(mat_elem_t));
  for (size_t i = 0; i < n; i++) {
    out->data[i * n + p->data[i]] = 1.0f;
  }

  // Forward substitution: solve L * Y = out in-place
  // L is unit lower triangular (stored in strictly lower part of LU)
  for (size_t j = 0; j < n; j++) {
    const mat_elem_t *Lj = &LU[j * n];  // Column j of L

    // Update all columns of out: out[j+1:n, :] -= out[j, :] * L[j+1:n, j]
    for (size_t col = 0; col < n; col++) {
      mat_elem_t *out_col = &out->data[col * n];
      mat_elem_t out_jcol = out_col[j];
      if (out_jcol == 0) continue;

      for (size_t i = j + 1; i < n; i++) {
        out_col[i] -= out_jcol * Lj[i];
      }
    }
  }

  // Back substitution: solve U * X = Y in-place
  // U is upper triangular (stored in upper part of LU including diagonal)
  for (size_t jj = n; jj > 0; jj--) {
    size_t j = jj - 1;
    const mat_elem_t *Uj = &LU[j * n];  // Column j of U
    mat_elem_t diag_inv = 1.0f / Uj[j];

    // Scale row j and update rows above
    for (size_t col = 0; col < n; col++) {
      mat_elem_t *out_col = &out->data[col * n];
      out_col[j] *= diag_inv;
      mat_elem_t out_jcol = out_col[j];

      for (size_t i = 0; i < j; i++) {
        out_col[i] -= out_jcol * Uj[i];
      }
    }
  }

  MAT_FREE_MAT(M);
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

/* SVD */

// Column-major SVD helpers - columns are contiguous for efficient SIMD
// W->data is column-major: W->data[row + col*m] = W[row, col]

/* Bidiagonalization */

// Apply Householder from left to zero out column below diagonal
// H = I - tau * v * v^T, applied as A = H * A = A - tau * v * (v^T * A)
MAT_INTERNAL_STATIC void mat__bidiag_left_(mat_elem_t *A, size_t m, size_t n,
                                           size_t lda, size_t k,
                                           mat_elem_t *v, mat_elem_t tau) {
  if (tau == 0) return;

  // For each column j >= k: A[:,j] -= tau * v * (v^T * A[:,j])
  for (size_t j = k; j < n; j++) {
    mat_elem_t *col_j = &A[j * lda];
    mat_elem_t dot = mat__dot_kernel(&col_j[k], v, m - k);
    mat__axpy_kernel(&col_j[k], -tau * dot, v, m - k);
  }
}

// Apply Householder from right to zero out row after superdiagonal
// H = I - tau * v * v^T, applied as A = A * H = A - tau * (A * v) * v^T
// A * v = sum_j v[j] * A[:,k+1+j]  (for rows k:m)
MAT_INTERNAL_STATIC void mat__bidiag_right_(mat_elem_t *A, size_t m, size_t n,
                                            size_t lda, size_t k,
                                            mat_elem_t *v, mat_elem_t tau) {
  if (tau == 0) return;

  size_t vlen = n - k - 1;
  size_t row_len = m - k;

  // Compute w = A[k:m, k+1:n] * v = sum_j v[j] * A[k:m, k+1+j]
  mat_elem_t *w = (mat_elem_t *)MAT_MALLOC(row_len * sizeof(mat_elem_t));
  memset(w, 0, row_len * sizeof(mat_elem_t));
  for (size_t j = 0; j < vlen; j++) {
    mat__axpy_kernel(w, v[j], &A[(k + 1 + j) * lda + k], row_len);
  }

  // A[k:m, k+1+j] -= tau * v[j] * w for each j
  for (size_t j = 0; j < vlen; j++) {
    mat__axpy_kernel(&A[(k + 1 + j) * lda + k], -tau * v[j], w, row_len);
  }

  MAT_FREE(w);
}

// Bidiagonalize matrix A (m x n, m >= n) in place
// Returns diagonal d[n] and superdiagonal e[n-1]
// Optionally accumulates U (m x m) and V (n x n) if non-NULL
// A is column-major with leading dimension lda
MAT_INTERNAL_STATIC void mat__bidiag_(mat_elem_t *A, size_t m, size_t n,
                                      size_t lda, mat_elem_t *d, mat_elem_t *e,
                                      mat_elem_t *U, mat_elem_t *V) {
  // Workspace for Householder vectors
  mat_elem_t *v_left = (mat_elem_t *)MAT_MALLOC(m * sizeof(mat_elem_t));
  mat_elem_t *v_right = (mat_elem_t *)MAT_MALLOC(n * sizeof(mat_elem_t));

  // Initialize U and V to identity if provided
  if (U) {
    memset(U, 0, m * m * sizeof(mat_elem_t));
    for (size_t i = 0; i < m; i++) U[i * m + i] = 1;
  }
  if (V) {
    memset(V, 0, n * n * sizeof(mat_elem_t));
    for (size_t i = 0; i < n; i++) V[i * n + i] = 1;
  }

  for (size_t k = 0; k < n; k++) {
    // Left Householder: zero out A[k+1:m, k]
    if (k < m - 1) {
      mat_elem_t *col_k = &A[k * lda];
      size_t vlen = m - k;

      // Copy column to v_left and compute Householder
      mat__copy_kernel(v_left, &col_k[k], vlen);

      mat_elem_t norm = MAT_SQRT(mat__dot_kernel(v_left, v_left, vlen));
      mat_elem_t tau_l = 0;

      if (norm > MAT_DEFAULT_EPSILON) {
        mat_elem_t x0 = v_left[0];
        mat_elem_t beta = (x0 >= 0) ? -norm : norm;
        tau_l = (beta - x0) / beta;
        mat_elem_t scale = 1.0f / (x0 - beta);
        for (size_t i = 1; i < vlen; i++) v_left[i] *= scale;
        v_left[0] = 1;
        d[k] = beta;  // Diagonal element
      } else {
        d[k] = col_k[k];
      }

      // Apply to A (columns k to n-1)
      mat__bidiag_left_(A, m, n, lda, k, v_left, tau_l);
      col_k[k] = d[k];  // Restore diagonal

      // Accumulate U: U = U * H_l where H_l = I - tau * v * v^T
      // (U * H)[:,j] = U[:,j] - tau * v[j] * (U * v)
      // U * v = sum_r v[r] * U[:,k+r]  (linear combination of columns)
      if (U && tau_l != 0) {
        mat_elem_t *w = (mat_elem_t *)MAT_MALLOC(m * sizeof(mat_elem_t));
        memset(w, 0, m * sizeof(mat_elem_t));
        // w = sum_r v[r] * U[:,k+r]
        for (size_t r = 0; r < vlen; r++) {
          mat__axpy_kernel(w, v_left[r], &U[(k + r) * m], m);
        }
        // U[:,j] -= tau * v[j-k] * w for j in [k, m)
        for (size_t j = k; j < m; j++) {
          mat__axpy_kernel(&U[j * m], -tau_l * v_left[j - k], w, m);
        }
        MAT_FREE(w);
      }
    } else {
      d[k] = A[k * lda + k];
    }

    // Right Householder: zero out A[k, k+2:n]
    if (k < n - 2) {
      size_t vlen = n - k - 1;

      // Extract row k, columns k+1 to n-1
      for (size_t j = 0; j < vlen; j++) {
        v_right[j] = A[(k + 1 + j) * lda + k];
      }

      mat_elem_t norm = MAT_SQRT(mat__dot_kernel(v_right, v_right, vlen));
      mat_elem_t tau_r = 0;

      if (norm > MAT_DEFAULT_EPSILON) {
        mat_elem_t x0 = v_right[0];
        mat_elem_t beta = (x0 >= 0) ? -norm : norm;
        tau_r = (beta - x0) / beta;
        mat_elem_t scale = 1.0f / (x0 - beta);
        for (size_t i = 1; i < vlen; i++) v_right[i] *= scale;
        v_right[0] = 1;
        e[k] = beta;  // Superdiagonal element
      } else {
        e[k] = A[(k + 1) * lda + k];
      }

      // Apply to A (rows k to m-1)
      mat__bidiag_right_(A, m, n, lda, k, v_right, tau_r);
      A[(k + 1) * lda + k] = e[k];  // Restore superdiagonal

      // Accumulate V: V = V * H_r where H_r = I - tau * v * v^T
      // V * v = sum_r v[r] * V[:,k+1+r]  (linear combination of columns)
      if (V && tau_r != 0) {
        mat_elem_t *w = (mat_elem_t *)MAT_MALLOC(n * sizeof(mat_elem_t));
        memset(w, 0, n * sizeof(mat_elem_t));
        // w = sum_r v[r] * V[:,k+1+r]
        for (size_t r = 0; r < vlen; r++) {
          mat__axpy_kernel(w, v_right[r], &V[(k + 1 + r) * n], n);
        }
        // V[:,j] -= tau * v[j-(k+1)] * w for j in [k+1, n)
        for (size_t j = k + 1; j < n; j++) {
          mat__axpy_kernel(&V[j * n], -tau_r * v_right[j - (k + 1)], w, n);
        }
        MAT_FREE(w);
      }
    } else if (k < n - 1) {
      e[k] = A[(k + 1) * lda + k];
    }
  }

  MAT_FREE(v_left);
  MAT_FREE(v_right);
}

/* Bidiagonal QR Iteration */

// Compute Givens rotation to zero out b given [a; b]
// Returns c, s such that [c s; -s c]^T * [a; b] = [r; 0]
MAT_INTERNAL_STATIC void mat__givens_(mat_elem_t a, mat_elem_t b,
                                      mat_elem_t *c, mat_elem_t *s) {
  if (b == 0) {
    *c = (a >= 0) ? 1 : -1;
    *s = 0;
  } else if (a == 0) {
    *c = 0;
    *s = (b >= 0) ? 1 : -1;
  } else if (MAT_FABS(b) > MAT_FABS(a)) {
    mat_elem_t t = a / b;
    mat_elem_t u = MAT_SQRT(1 + t * t);
    if (b < 0) u = -u;
    *s = 1 / u;
    *c = (*s) * t;
  } else {
    mat_elem_t t = b / a;
    mat_elem_t u = MAT_SQRT(1 + t * t);
    if (a < 0) u = -u;
    *c = 1 / u;
    *s = (*c) * t;
  }
}

// Apply Givens rotation to columns i, j of matrix
// [col_i col_j] = [col_i col_j] * [c -s; s c]

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void mat__givens_cols_neon_(mat_elem_t *A, size_t ld,
                                                size_t nrows, size_t i, size_t j,
                                                mat_elem_t c, mat_elem_t s) {
  mat_elem_t *col_i = &A[i * ld];
  mat_elem_t *col_j = &A[j * ld];

  MAT_NEON_TYPE vc = MAT_NEON_DUP(c);
  MAT_NEON_TYPE vs = MAT_NEON_DUP(s);
  MAT_NEON_TYPE vns = MAT_NEON_DUP(-s);

  size_t k = 0;

  // 4x unrolled loop
  for (; k + MAT_NEON_WIDTH * 4 <= nrows; k += MAT_NEON_WIDTH * 4) {
    MAT_NEON_TYPE vi0 = MAT_NEON_LOAD(&col_i[k]);
    MAT_NEON_TYPE vi1 = MAT_NEON_LOAD(&col_i[k + MAT_NEON_WIDTH]);
    MAT_NEON_TYPE vi2 = MAT_NEON_LOAD(&col_i[k + MAT_NEON_WIDTH * 2]);
    MAT_NEON_TYPE vi3 = MAT_NEON_LOAD(&col_i[k + MAT_NEON_WIDTH * 3]);
    MAT_NEON_TYPE vj0 = MAT_NEON_LOAD(&col_j[k]);
    MAT_NEON_TYPE vj1 = MAT_NEON_LOAD(&col_j[k + MAT_NEON_WIDTH]);
    MAT_NEON_TYPE vj2 = MAT_NEON_LOAD(&col_j[k + MAT_NEON_WIDTH * 2]);
    MAT_NEON_TYPE vj3 = MAT_NEON_LOAD(&col_j[k + MAT_NEON_WIDTH * 3]);

    // new_i = c * ci + s * cj
    MAT_NEON_TYPE ni0 = MAT_NEON_FMA(MAT_NEON_MUL(vc, vi0), vj0, vs);
    MAT_NEON_TYPE ni1 = MAT_NEON_FMA(MAT_NEON_MUL(vc, vi1), vj1, vs);
    MAT_NEON_TYPE ni2 = MAT_NEON_FMA(MAT_NEON_MUL(vc, vi2), vj2, vs);
    MAT_NEON_TYPE ni3 = MAT_NEON_FMA(MAT_NEON_MUL(vc, vi3), vj3, vs);

    // new_j = c * cj - s * ci = c * cj + (-s) * ci
    MAT_NEON_TYPE nj0 = MAT_NEON_FMA(MAT_NEON_MUL(vc, vj0), vi0, vns);
    MAT_NEON_TYPE nj1 = MAT_NEON_FMA(MAT_NEON_MUL(vc, vj1), vi1, vns);
    MAT_NEON_TYPE nj2 = MAT_NEON_FMA(MAT_NEON_MUL(vc, vj2), vi2, vns);
    MAT_NEON_TYPE nj3 = MAT_NEON_FMA(MAT_NEON_MUL(vc, vj3), vi3, vns);

    MAT_NEON_STORE(&col_i[k], ni0);
    MAT_NEON_STORE(&col_i[k + MAT_NEON_WIDTH], ni1);
    MAT_NEON_STORE(&col_i[k + MAT_NEON_WIDTH * 2], ni2);
    MAT_NEON_STORE(&col_i[k + MAT_NEON_WIDTH * 3], ni3);
    MAT_NEON_STORE(&col_j[k], nj0);
    MAT_NEON_STORE(&col_j[k + MAT_NEON_WIDTH], nj1);
    MAT_NEON_STORE(&col_j[k + MAT_NEON_WIDTH * 2], nj2);
    MAT_NEON_STORE(&col_j[k + MAT_NEON_WIDTH * 3], nj3);
  }

  // Single-width loop
  for (; k + MAT_NEON_WIDTH <= nrows; k += MAT_NEON_WIDTH) {
    MAT_NEON_TYPE vi = MAT_NEON_LOAD(&col_i[k]);
    MAT_NEON_TYPE vj = MAT_NEON_LOAD(&col_j[k]);
    MAT_NEON_TYPE ni = MAT_NEON_FMA(MAT_NEON_MUL(vc, vi), vj, vs);
    MAT_NEON_TYPE nj = MAT_NEON_FMA(MAT_NEON_MUL(vc, vj), vi, vns);
    MAT_NEON_STORE(&col_i[k], ni);
    MAT_NEON_STORE(&col_j[k], nj);
  }

  // Scalar cleanup
  for (; k < nrows; k++) {
    mat_elem_t ci = col_i[k];
    mat_elem_t cj = col_j[k];
    col_i[k] = c * ci + s * cj;
    col_j[k] = -s * ci + c * cj;
  }
}
#endif

MAT_INTERNAL_STATIC void mat__givens_cols_scalar_(mat_elem_t *A, size_t ld,
                                                  size_t nrows, size_t i,
                                                  size_t j, mat_elem_t c,
                                                  mat_elem_t s) {
  mat_elem_t *col_i = &A[i * ld];
  mat_elem_t *col_j = &A[j * ld];
  for (size_t k = 0; k < nrows; k++) {
    mat_elem_t ci = col_i[k];
    mat_elem_t cj = col_j[k];
    col_i[k] = c * ci + s * cj;
    col_j[k] = -s * ci + c * cj;
  }
}

MAT_INTERNAL_STATIC void mat__givens_cols_(mat_elem_t *A, size_t ld, size_t nrows,
                                           size_t i, size_t j,
                                           mat_elem_t c, mat_elem_t s) {
#if defined(MAT_HAS_ARM_NEON)
  // Need enough rows for NEON to be beneficial
  if (nrows >= MAT_NEON_WIDTH * 4) {
    mat__givens_cols_neon_(A, ld, nrows, i, j, c, s);
  } else {
    mat__givens_cols_scalar_(A, ld, nrows, i, j, c, s);
  }
#else
  mat__givens_cols_scalar_(A, ld, nrows, i, j, c, s);
#endif
}

// Apply Givens rotation to rows i, j of matrix (strided access)
// [row_i; row_j] = [c s; -s c] * [row_i; row_j]
// Note: Strided access pattern doesn't benefit from SIMD
MAT_INTERNAL_STATIC void mat__givens_rows_(mat_elem_t *A, size_t ld,
                                           size_t row_i, size_t row_j,
                                           size_t col_start, size_t col_end,
                                           mat_elem_t c, mat_elem_t s) {
  for (size_t col = col_start; col < col_end; col++) {
    mat_elem_t ti = A[col * ld + row_i];
    mat_elem_t tj = A[col * ld + row_j];
    A[col * ld + row_i] = c * ti + s * tj;
    A[col * ld + row_j] = -s * ti + c * tj;
  }
}

// One QR iteration step on bidiagonal matrix (Golub-Kahan SVD step)
// Works on submatrix from index p to q (inclusive)
// d[p:q+1] is diagonal, e[p:q] is superdiagonal
MAT_INTERNAL_STATIC void mat__svd_qr_step_(mat_elem_t *d, mat_elem_t *e,
                                           size_t p, size_t q,
                                           mat_elem_t *U, size_t m,
                                           mat_elem_t *V, size_t n) {
  // Wilkinson shift from trailing 2x2 of T = B^T * B
  // T[n-1,n-1] = d[q]^2 + e[q-1]^2 (if q > p)
  // T[n-2,n-1] = d[q-1] * e[q-1]
  // T[n-2,n-2] = d[q-1]^2 + e[q-2]^2 (if q > p+1)
  mat_elem_t e_qm1 = (q > p) ? e[q - 1] : 0;
  mat_elem_t t_nn = d[q] * d[q] + e_qm1 * e_qm1;
  mat_elem_t t_nm1n = (q > p) ? d[q - 1] * e[q - 1] : 0;
  mat_elem_t e_qm2 = (q > p + 1) ? e[q - 2] : 0;
  mat_elem_t t_nm1nm1 = (q > p) ? (d[q - 1] * d[q - 1] + e_qm2 * e_qm2) : 0;

  mat_elem_t delta = (t_nm1nm1 - t_nn) / 2;
  mat_elem_t shift;
  if (delta == 0 && t_nm1n == 0) {
    shift = 0;
  } else if (delta == 0) {
    shift = t_nn - MAT_FABS(t_nm1n);
  } else {
    mat_elem_t sign = (delta >= 0) ? 1 : -1;
    shift = t_nn - t_nm1n * t_nm1n / (delta + sign * MAT_SQRT(delta * delta + t_nm1n * t_nm1n));
  }

  // Chase the bulge
  mat_elem_t f = d[p] * d[p] - shift;
  mat_elem_t g = d[p] * e[p];

  for (size_t k = p; k < q; k++) {
    mat_elem_t c, s;

    // Right Givens rotation to zero g
    mat__givens_(f, g, &c, &s);

    if (k > p) e[k - 1] = f * c + g * s;  // This should equal the computed r

    // Apply to columns k, k+1 of B:
    // [d[k] e[k]] [c  s]   [c*d[k]+s*e[k]   -s*d[k]+c*e[k]]
    // [0    d[k+1]] [-s c] = [s*d[k+1]        c*d[k+1]      ]
    mat_elem_t dk = d[k], ek = e[k], dk1 = d[k + 1];
    mat_elem_t new_dk = c * dk + s * ek;
    mat_elem_t new_ek = -s * dk + c * ek;
    mat_elem_t bulge = s * dk1;
    mat_elem_t new_dk1 = c * dk1;

    // Accumulate V
    if (V) mat__givens_cols_(V, n, n, k, k + 1, c, s);

    // Left Givens rotation to zero bulge
    mat__givens_(new_dk, bulge, &c, &s);

    // Apply to rows k, k+1 of B:
    d[k] = c * new_dk + s * bulge;  // = r from Givens
    // [new_ek  new_dk1]   [c -s]   [c*new_ek+s*new_dk1  ...]
    // [0       e[k+1] ] * [s  c] = [s*e[k+1]            ...]
    mat_elem_t next_ek = c * new_ek + s * new_dk1;
    d[k + 1] = -s * new_ek + c * new_dk1;
    e[k] = next_ek;

    // Accumulate U
    if (U) mat__givens_cols_(U, m, m, k, k + 1, c, s);

    if (k + 1 < q) {
      // Bulge appears at e[k+1] position
      mat_elem_t old_ek1 = e[k + 1];
      f = e[k];
      g = s * old_ek1;
      e[k + 1] = c * old_ek1;
    }
  }
}

// QR iteration for bidiagonal SVD
// Input: d[n] diagonal, e[n-1] superdiagonal
// Output: d[n] contains singular values (may be negative, need abs)
// U (m x m) and V (n x n) accumulate rotations if non-NULL
MAT_INTERNAL_STATIC void mat__svd_qr_bidiag_(mat_elem_t *d, mat_elem_t *e, size_t n,
                                            mat_elem_t *U, size_t m,
                                            mat_elem_t *V) {
  const int max_iters = 30 * (int)n;
  const mat_elem_t tol = MAT_DEFAULT_EPSILON;

  size_t q = n - 1;  // End of active submatrix
  int iter = 0;

  while (q > 0 && iter < max_iters) {
    // Check for negligible e[q-1] (convergence)
    mat_elem_t thresh = tol * (MAT_FABS(d[q - 1]) + MAT_FABS(d[q]));
    if (MAT_FABS(e[q - 1]) <= thresh) {
      e[q - 1] = 0;
      q--;
      continue;
    }

    // Find start of active submatrix (p)
    size_t p = q - 1;
    while (p > 0) {
      thresh = tol * (MAT_FABS(d[p - 1]) + MAT_FABS(d[p]));
      if (MAT_FABS(e[p - 1]) <= thresh) {
        e[p - 1] = 0;
        break;
      }
      p--;
    }

    // Check for zero diagonal in active submatrix
    int found_zero = 0;
    for (size_t k = p; k <= q; k++) {
      if (MAT_FABS(d[k]) < tol) {
        // Zero out row k by chasing with Givens rotations
        d[k] = 0;
        if (k < q) {
          for (size_t j = k; j < q; j++) {
            mat_elem_t c, s;
            mat__givens_(d[j + 1], e[j], &c, &s);
            d[j + 1] = c * d[j + 1] + s * e[j];
            if (j + 1 < q) {
              mat_elem_t tmp = -s * e[j + 1];
              e[j + 1] = c * e[j + 1];
              e[j] = tmp;
            } else {
              e[j] = 0;
            }
            if (U) mat__givens_cols_(U, m, m, k, j + 1, c, s);
          }
        }
        found_zero = 1;
        break;
      }
    }

    if (found_zero) continue;

    // Perform one QR step
    mat__svd_qr_step_(d, e, p, q, U, m, V, n);
    iter++;
  }

  // Make singular values positive
  for (size_t i = 0; i < n; i++) {
    if (d[i] < 0) {
      d[i] = -d[i];
      // Flip sign of corresponding V column
      if (V) {
        for (size_t j = 0; j < n; j++) {
          V[i * n + j] = -V[i * n + j];
        }
      }
    }
  }
}

// Complete bidiagonalization + QR iteration SVD
// Input: A (m x n, m >= n), column-major
// Output: U (m x m), S (n singular values), Vt (n x n)
MAT_INTERNAL_STATIC void mat__svd_bidiag_qr_(const Mat *A, Mat *U, Vec *S, Mat *Vt) {
  size_t m = A->rows;
  size_t n = A->cols;

  // Workspace
  mat_elem_t *B = (mat_elem_t *)MAT_MALLOC(m * n * sizeof(mat_elem_t));
  mat_elem_t *d = (mat_elem_t *)MAT_MALLOC(n * sizeof(mat_elem_t));
  mat_elem_t *e = (mat_elem_t *)MAT_MALLOC(n * sizeof(mat_elem_t));
  mat_elem_t *U_b = (mat_elem_t *)MAT_MALLOC(m * m * sizeof(mat_elem_t));
  mat_elem_t *V_b = (mat_elem_t *)MAT_MALLOC(n * n * sizeof(mat_elem_t));
  mat_elem_t *U_qr = (mat_elem_t *)MAT_MALLOC(m * m * sizeof(mat_elem_t));
  mat_elem_t *V_qr = (mat_elem_t *)MAT_MALLOC(n * n * sizeof(mat_elem_t));

  // Copy A to workspace (column-major)
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++) {
      B[j * m + i] = MAT_AT(A, i, j);
    }
  }

  // Step 1: Bidiagonalize
  mat__bidiag_(B, m, n, m, d, e, U_b, V_b);

  // Step 2: Initialize U_qr, V_qr to identity
  memset(U_qr, 0, m * m * sizeof(mat_elem_t));
  memset(V_qr, 0, n * n * sizeof(mat_elem_t));
  for (size_t i = 0; i < m; i++) U_qr[i * m + i] = 1;
  for (size_t i = 0; i < n; i++) V_qr[i * n + i] = 1;

  // Step 3: QR iteration
  mat__svd_qr_bidiag_(d, e, n, U_qr, m, V_qr);

  // Step 4: Sort singular values descending and track permutation
  size_t *perm = (size_t *)MAT_MALLOC(n * sizeof(size_t));
  for (size_t i = 0; i < n; i++) perm[i] = i;
  for (size_t i = 0; i < n - 1; i++) {
    for (size_t j = i + 1; j < n; j++) {
      if (d[perm[j]] > d[perm[i]]) {
        size_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
      }
    }
  }

  // Step 5: Reorder U_qr columns according to perm, then compute U = U_b * U_qr
  // First permute columns of U_qr in place for the first n columns
  mat_elem_t *U_qr_perm = (mat_elem_t *)MAT_MALLOC(m * m * sizeof(mat_elem_t));
  for (size_t j = 0; j < m; j++) {
    size_t src_col = (j < n) ? perm[j] : j;
    mat__copy_kernel(&U_qr_perm[j * m], &U_qr[src_col * m], m);
  }

  // U = U_b * U_qr_perm using optimized GEMM
  Mat U_b_mat = {.rows = m, .cols = m, .data = U_b};
  Mat U_qr_mat = {.rows = m, .cols = m, .data = U_qr_perm};
  mat_gemm(U, 1, &U_b_mat, &U_qr_mat, 0);
  MAT_FREE(U_qr_perm);

  // Step 6: Reorder V_qr columns according to perm, compute V = V_b * V_qr, output Vt
  mat_elem_t *V_qr_perm = (mat_elem_t *)MAT_MALLOC(n * n * sizeof(mat_elem_t));
  for (size_t j = 0; j < n; j++) {
    mat__copy_kernel(&V_qr_perm[j * n], &V_qr[perm[j] * n], n);
  }

  // V_temp = V_b * V_qr_perm, then Vt = V_temp^T
  mat_elem_t *V_temp = (mat_elem_t *)MAT_MALLOC(n * n * sizeof(mat_elem_t));
  Mat V_b_mat = {.rows = n, .cols = n, .data = V_b};
  Mat V_qr_mat = {.rows = n, .cols = n, .data = V_qr_perm};
  Mat V_temp_mat = {.rows = n, .cols = n, .data = V_temp};
  mat_gemm(&V_temp_mat, 1, &V_b_mat, &V_qr_mat, 0);

  // Transpose V_temp to get Vt
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      MAT_SET(Vt, i, j, V_temp[i * n + j]);  // V_temp is col-major: V[i,j] = V_temp[j*n+i]
    }
  }
  MAT_FREE(V_qr_perm);
  MAT_FREE(V_temp);

  // Step 7: Copy sorted singular values to S
  for (size_t i = 0; i < n; i++) {
    S->data[i] = d[perm[i]];
  }

  MAT_FREE(B); MAT_FREE(d); MAT_FREE(e);
  MAT_FREE(U_b); MAT_FREE(V_b);
  MAT_FREE(U_qr); MAT_FREE(V_qr);
  MAT_FREE(perm);
}

// Apply Jacobi rotation to columns i and j (column-major data)
// new_i = cs * col_i - sn * col_j
// new_j = sn * col_i + cs * col_j
MAT_INTERNAL_STATIC void mat__svd_rotate_cols_(mat_elem_t *data, size_t m,
                                               size_t i, size_t j,
                                               mat_elem_t cs, mat_elem_t sn) {
  mat_elem_t *col_i = &data[i * m];
  mat_elem_t *col_j = &data[j * m];

#ifdef MAT_HAS_ARM_NEON
  MAT_NEON_TYPE vcs = MAT_NEON_DUP(cs);
  MAT_NEON_TYPE vsn = MAT_NEON_DUP(sn);

  size_t k = 0;
  for (; k + MAT_NEON_WIDTH <= m; k += MAT_NEON_WIDTH) {
    MAT_NEON_TYPE wi = MAT_NEON_LOAD(&col_i[k]);
    MAT_NEON_TYPE wj = MAT_NEON_LOAD(&col_j[k]);

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
#else
  for (size_t k = 0; k < m; k++) {
    mat_elem_t wi = col_i[k];
    mat_elem_t wj = col_j[k];
    col_i[k] = cs * wi - sn * wj;
    col_j[k] = sn * wi + cs * wj;
  }
#endif
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
MAT_INTERNAL_STATIC void mat__svd_axpy_strided_(Vec *y, mat_elem_t alpha,
                                                const Mat *X, size_t col) {
  const mat_elem_t *x = &X->data[col];
  size_t x_stride = X->cols;
  size_t n = X->rows;
  mat_elem_t *yp = y->data;

#ifdef MAT_HAS_ARM_NEON
  MAT_NEON_TYPE valpha = MAT_NEON_DUP(alpha);
  size_t i = 0;

  for (; i + MAT_NEON_WIDTH <= n; i += MAT_NEON_WIDTH) {
    MAT_NEON_TYPE vy = MAT_NEON_LOAD(&yp[i]);
#ifdef MAT_DOUBLE_PRECISION
    MAT_NEON_TYPE vx = {x[i * x_stride], x[(i + 1) * x_stride]};
#else
    MAT_NEON_TYPE vx = {x[i * x_stride], x[(i + 1) * x_stride],
                        x[(i + 2) * x_stride], x[(i + 3) * x_stride]};
#endif
    vy = MAT_NEON_FMA(vy, vx, valpha);
    MAT_NEON_STORE(&yp[i], vy);
  }

  for (; i < n; i++) {
    yp[i] += alpha * x[i * x_stride];
  }
#else
  for (size_t i = 0; i < n; i++) {
    yp[i] += alpha * x[i * x_stride];
  }
#endif
}

// dot(y, X[:,col]) where y is contiguous Vec and X is row-major Mat
MAT_INTERNAL_STATIC mat_elem_t mat__svd_dot_strided_(const Vec *y,
                                                     const Mat *X, size_t col) {
  const mat_elem_t *yp = y->data;
  const mat_elem_t *x = &X->data[col];
  size_t x_stride = X->cols;
  size_t n = X->rows;

#ifdef MAT_HAS_ARM_NEON
  MAT_NEON_TYPE vsum0 = MAT_NEON_DUP(0);
  MAT_NEON_TYPE vsum1 = MAT_NEON_DUP(0);
  size_t i = 0;

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

  for (; i < n; i++) {
    sum += yp[i] * x[i * x_stride];
  }
  return sum;
#else
  mat_elem_t sum = 0;
  for (size_t i = 0; i < n; i++) {
    sum += yp[i] * x[i * x_stride];
  }
  return sum;
#endif
}

// Max column L2 norm for column-major matrix
// Returns max_j ||data[:,j]||_2 where column j is at data[j*col_len]
MAT_INTERNAL_STATIC mat_elem_t mat__max_col_norm_(const mat_elem_t *data,
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
MAT_INTERNAL_STATIC void mat__svd_jacobi_iter_(mat_elem_t *W_data, size_t m,
                                               mat_elem_t *V_data, size_t n,
                                               size_t col_stride) {
  const int max_sweeps = 30;
  const mat_elem_t tol = MAT_DEFAULT_EPSILON;

  mat_elem_t *col_norms = (mat_elem_t *)MAT_MALLOC(n * sizeof(mat_elem_t));

  // Compute initial column norms (columns are contiguous)
  for (size_t j = 0; j < n; j++) {
    mat_elem_t *col_j = &W_data[j * col_stride];
    col_norms[j] = mat__dot_kernel(col_j, col_j, m);
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
        mat_elem_t c = mat__dot_kernel(col_i, col_j, m);

        if (MAT_FABS(c) < tol * MAT_SQRT(a * b + tol))
          continue;

        rotations++;
        mat_elem_t cs, sn;
        mat_svd_jacobi_rotation(a, b, c, &cs, &sn);

        mat__svd_rotate_cols_(W_data, col_stride, i, j, cs, sn);
        mat__svd_rotate_cols_(V_data, n, i, j, cs, sn);

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
  // Use Bidiag+QR for larger matrices (faster), Jacobi for smaller (simpler)
#define MAT_SVD_BIDIAG_THRESHOLD 20
  if (n >= MAT_SVD_BIDIAG_THRESHOLD) {
    mat__svd_bidiag_qr_(A, U, S, Vt);
    return;
  }

  // Jacobi SVD for small matrices
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
  mat__svd_jacobi_iter_(W_data, m, V_data, n, m);

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
  mat__svd_jacobi_iter_(W_data, m, V_data, n, m);

  // Tolerance based on max singular value
  mat_elem_t max_sigma = mat__max_col_norm_(W_data, m, n);
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

/* Eigenvalues (QR Algorithm) */

// Reduce matrix to upper Hessenberg form using Householder reflections
// H = Q^T * A * Q where H is upper Hessenberg (zeros below sub-diagonal)
// Modifies A in-place to become H
// Reduce matrix A to upper Hessenberg form using Householder reflections.
// If Q is provided (not NULL), accumulate the orthogonal transformation
// Q = H_1 * H_2 * ... * H_{n-2}. Q must be initialized to identity before calling.
MAT_INTERNAL_STATIC void mat__hessenberg_(Mat *A, mat_elem_t *Q, size_t ldq) {
  size_t n = A->rows;
  if (n <= 2)
    return;

  mat_elem_t *v_data = (mat_elem_t *)MAT_MALLOC(n * sizeof(mat_elem_t));
  mat_elem_t *w_data = (mat_elem_t *)MAT_MALLOC(n * sizeof(mat_elem_t));

  for (size_t k = 0; k < n - 2; k++) {
    size_t len = n - k - 1;

    // Column k starts at A->data[k * n], subcolumn at row k+1
    mat_elem_t *col_k = &A->data[k * n + (k + 1)];

    // Copy subcolumn to v (contiguous in column-major)
    memcpy(v_data, col_k, len * sizeof(mat_elem_t));

    Vec v_sub = {.data = v_data, .rows = len, .cols = 1};

    // Compute Householder reflection
    mat_elem_t tau;
    mat_elem_t beta = mat_householder(&v_sub, &tau, &v_sub);

    if (MAT_FABS(tau) < MAT_DEFAULT_EPSILON)
      continue;

    // Set sub-diagonal element and zeros below
    col_k[0] = beta;
    memset(&col_k[1], 0, (len - 1) * sizeof(mat_elem_t));

    // Apply from left: A[k+1:n, k+1:n] = H * A[k+1:n, k+1:n]
    // For each column j: A[:,j] -= tau * v * (v^T * A[:,j])
    for (size_t j = k + 1; j < n; j++) {
      mat_elem_t *col_j = &A->data[j * n + (k + 1)];
      mat_elem_t dot = mat__dot_kernel(v_data, col_j, len);
      mat__axpy_kernel(col_j, -tau * dot, v_data, len);
    }

    // Apply from right: A[0:n, k+1:n] = A[0:n, k+1:n] * H
    // w = A * v, then A -= tau * w * v^T
    // Compute w[i] = sum_j A[i, k+1+j] * v[j] for each row i
    memset(w_data, 0, n * sizeof(mat_elem_t));
    for (size_t j = 0; j < len; j++) {
      mat_elem_t *col_j = &A->data[(k + 1 + j) * n];
      mat__axpy_kernel(w_data, v_data[j], col_j, n);
    }

    // A[:,k+1+j] -= tau * w * v[j]
    for (size_t j = 0; j < len; j++) {
      mat_elem_t *col_j = &A->data[(k + 1 + j) * n];
      mat__axpy_kernel(col_j, -tau * v_data[j], w_data, n);
    }

    // Accumulate Q = Q * H_k if requested
    // H_k = I - tau * v * v^T operates on rows/cols k+1:n
    // Q * H_k = Q - tau * (Q * v) * v^T
    // Only columns k+1:n of Q are affected
    if (Q) {
      // w = Q[:, k+1:n] * v (size n vector)
      memset(w_data, 0, n * sizeof(mat_elem_t));
      for (size_t j = 0; j < len; j++) {
        mat_elem_t *Q_col = &Q[(k + 1 + j) * ldq];
        mat__axpy_kernel(w_data, v_data[j], Q_col, n);
      }
      // Q[:, k+1+j] -= tau * w * v[j]
      for (size_t j = 0; j < len; j++) {
        mat_elem_t *Q_col = &Q[(k + 1 + j) * ldq];
        mat__axpy_kernel(Q_col, -tau * v_data[j], w_data, n);
      }
    }
  }

  MAT_FREE(v_data);
  MAT_FREE(w_data);
}

// Francis double-shift QR step on Hessenberg matrix H
// Uses implicit double shift for better convergence with complex eigenvalues
// Reference: Golub & Van Loan, "Matrix Computations", Algorithm 7.5.1
// If Z is provided, accumulate transformations: Z = Z * (product of rotations)
MAT_INTERNAL_STATIC void mat__qr_step_(Mat *H, size_t lo, size_t hi,
                                        mat_elem_t *Z, size_t ldz) {
  size_t n = H->rows;
  mat_elem_t *data = H->data;

  // For small blocks, use single-shift (simpler and sufficient)
  if (hi - lo < 2) {
    // 2x2 block: single Givens rotation
    mat_elem_t a = MAT_AT(H, lo, lo);
    mat_elem_t b = MAT_AT(H, lo, lo + 1);
    mat_elem_t cc = MAT_AT(H, lo + 1, lo);
    mat_elem_t d = MAT_AT(H, lo + 1, lo + 1);

    mat_elem_t trace = a + d;
    mat_elem_t det = a * d - b * cc;
    mat_elem_t disc = trace * trace - 4 * det;
    mat_elem_t shift = (disc >= 0) ?
      ((MAT_FABS((trace + MAT_SQRT(disc))/2 - d) < MAT_FABS((trace - MAT_SQRT(disc))/2 - d)) ?
       (trace + MAT_SQRT(disc))/2 : (trace - MAT_SQRT(disc))/2) : d;

    mat_elem_t x = a - shift;
    mat_elem_t y = cc;
    mat_elem_t c, s;
    mat__givens_(x, y, &c, &s);

    // Apply to rows lo, lo+1
    for (size_t j = lo; j < n; j++) {
      mat_elem_t *col = &data[j * n];
      mat_elem_t t0 = col[lo];
      mat_elem_t t1 = col[lo + 1];
      col[lo] = c * t0 + s * t1;
      col[lo + 1] = -s * t0 + c * t1;
    }
    // Apply to cols lo, lo+1
    mat__givens_cols_(data, n, hi + 1, lo, lo + 1, c, s);

    // Accumulate Z = Z * G^T
    if (Z) {
      mat__givens_cols_(Z, ldz, n, lo, lo + 1, c, s);
    }
    return;
  }

  // Francis double-shift: use trace and determinant of bottom 2x2
  mat_elem_t a = MAT_AT(H, hi - 1, hi - 1);
  mat_elem_t b = MAT_AT(H, hi - 1, hi);
  mat_elem_t cc = MAT_AT(H, hi, hi - 1);
  mat_elem_t d = MAT_AT(H, hi, hi);

  mat_elem_t s = a + d;  // trace = σ₁ + σ₂
  mat_elem_t t = a * d - b * cc;  // det = σ₁σ₂

  // First column of M = H² - sH + tI
  // M[0:3, 0] = H² e₁ - s H e₁ + t e₁
  // where e₁ = (1, 0, 0, ...)^T at position lo
  mat_elem_t h00 = MAT_AT(H, lo, lo);
  mat_elem_t h10 = MAT_AT(H, lo + 1, lo);
  mat_elem_t h01 = MAT_AT(H, lo, lo + 1);
  mat_elem_t h11 = MAT_AT(H, lo + 1, lo + 1);
  mat_elem_t h21 = MAT_AT(H, lo + 2, lo + 1);

  // First column of M = (H - σ₁I)(H - σ₂I)
  // x = h00² + h01*h10 - s*h00 + t
  // y = h10*(h00 + h11 - s)
  // z = h10*h21
  mat_elem_t x = h00 * h00 + h01 * h10 - s * h00 + t;
  mat_elem_t y = h10 * (h00 + h11 - s);
  mat_elem_t z = h10 * h21;

  // Chase 3x3 bulge down using Householder reflections
  for (size_t k = lo; k < hi - 1; k++) {
    // Compute Householder to zero y, z
    mat_elem_t norm_sq = x * x + y * y + z * z;
    mat_elem_t norm = MAT_SQRT(norm_sq);

    if (norm < MAT_DEFAULT_EPSILON) {
      // Skip if essentially zero
      if (k < hi - 2) {
        x = MAT_AT(H, k + 1, k);
        y = MAT_AT(H, k + 2, k);
        z = (k + 3 <= hi) ? MAT_AT(H, k + 3, k) : 0;
      }
      continue;
    }

    mat_elem_t beta = (x >= 0) ? -norm : norm;
    mat_elem_t v0 = x - beta;
    mat_elem_t tau = -v0 / beta;
    mat_elem_t inv_v0 = 1.0 / v0;
    mat_elem_t v1 = y * inv_v0;
    mat_elem_t v2 = z * inv_v0;

    // Determine the starting row
    size_t r0 = k;

    // Apply Householder from left: H[k:k+3, :] -= tau * v * (v^T * H[k:k+3, :])
    size_t col_start = (k > lo) ? k - 1 : lo;
    for (size_t j = col_start; j < n; j++) {
      mat_elem_t *col = &data[j * n];
      mat_elem_t dot = col[r0] + v1 * col[r0 + 1];
      if (r0 + 2 < n) dot += v2 * col[r0 + 2];

      col[r0] -= tau * dot;
      col[r0 + 1] -= tau * v1 * dot;
      if (r0 + 2 < n) col[r0 + 2] -= tau * v2 * dot;
    }

    // Apply Householder from right: H[:, k:k+3] -= tau * (H[:, k:k+3] * v) * v^T
    size_t row_end = (k + 4 < n) ? k + 4 : n;
    if (row_end > hi + 1) row_end = hi + 1;
    for (size_t i = 0; i < row_end; i++) {
      mat_elem_t hr0 = data[r0 * n + i];
      mat_elem_t hr1 = data[(r0 + 1) * n + i];
      mat_elem_t hr2 = (r0 + 2 < n) ? data[(r0 + 2) * n + i] : 0;

      mat_elem_t dot = hr0 + v1 * hr1 + v2 * hr2;

      data[r0 * n + i] -= tau * dot;
      data[(r0 + 1) * n + i] -= tau * v1 * dot;
      if (r0 + 2 < n) data[(r0 + 2) * n + i] -= tau * v2 * dot;
    }

    // Accumulate Z = Z * H_k where H_k = I - tau * v * v^T (v = [1, v1, v2])
    // Z * H_k = Z - tau * (Z * v) * v^T
    if (Z) {
      size_t rr0 = r0;  // r0 = k
      mat_elem_t *Z_k0 = &Z[rr0 * ldz];
      mat_elem_t *Z_k1 = &Z[(rr0 + 1) * ldz];
      mat_elem_t *Z_k2 = (rr0 + 2 < n) ? &Z[(rr0 + 2) * ldz] : NULL;
      for (size_t i = 0; i < n; i++) {
        // w[i] = Z[i, k] + v1 * Z[i, k+1] + v2 * Z[i, k+2]
        mat_elem_t w_i = Z_k0[i] + v1 * Z_k1[i];
        if (Z_k2) w_i += v2 * Z_k2[i];
        // Z[i, k] -= tau * w[i]
        // Z[i, k+1] -= tau * v1 * w[i]
        // Z[i, k+2] -= tau * v2 * w[i]
        Z_k0[i] -= tau * w_i;
        Z_k1[i] -= tau * v1 * w_i;
        if (Z_k2) Z_k2[i] -= tau * v2 * w_i;
      }
    }

    // Pick up bulge for next iteration
    if (k < hi - 2) {
      x = MAT_AT(H, k + 1, k);
      y = MAT_AT(H, k + 2, k);
      z = (k + 3 <= hi) ? MAT_AT(H, k + 3, k) : 0;
    }
  }

  // Final 2x2 Givens rotation at bottom
  mat_elem_t x_final = MAT_AT(H, hi - 1, hi - 2);
  mat_elem_t y_final = MAT_AT(H, hi, hi - 2);
  mat_elem_t c_final, s_final;
  mat__givens_(x_final, y_final, &c_final, &s_final);

  // Apply from left to rows hi-1, hi
  for (size_t j = hi - 2; j < n; j++) {
    mat_elem_t *col = &data[j * n];
    mat_elem_t t0 = col[hi - 1];
    mat_elem_t t1 = col[hi];
    col[hi - 1] = c_final * t0 + s_final * t1;
    col[hi] = -s_final * t0 + c_final * t1;
  }
  // Apply from right to cols hi-1, hi
  mat__givens_cols_(data, n, hi + 1, hi - 1, hi, c_final, s_final);

  // Accumulate final Givens into Z
  if (Z) {
    mat__givens_cols_(Z, ldz, n, hi - 1, hi, c_final, s_final);
  }
}

/* Multishift QR */

// Apply a 3x3 Householder reflector P = I - tau * v * v^T to H from both sides
// v = [1, v1, v2], affects rows/cols r0, r0+1, r0+2
// If Z is provided, accumulate: Z = Z * P
MAT_INTERNAL_STATIC void mat__apply_householder_3x3_(
    mat_elem_t *H, size_t n, size_t r0, size_t col_lo, size_t col_hi,
    size_t row_lo, size_t row_hi,
    mat_elem_t tau, mat_elem_t v1, mat_elem_t v2,
    mat_elem_t *Z, size_t ldz) {

  // Apply from left: H[r0:r0+3, col_lo:col_hi] = P * H[r0:r0+3, col_lo:col_hi]
  for (size_t j = col_lo; j <= col_hi; j++) {
    mat_elem_t *col = &H[j * n];
    mat_elem_t dot = col[r0] + v1 * col[r0 + 1] + v2 * col[r0 + 2];
    col[r0] -= tau * dot;
    col[r0 + 1] -= tau * v1 * dot;
    col[r0 + 2] -= tau * v2 * dot;
  }

  // Apply from right: H[row_lo:row_hi, r0:r0+3] = H[row_lo:row_hi, r0:r0+3] * P
  for (size_t i = row_lo; i <= row_hi; i++) {
    mat_elem_t hr0 = H[r0 * n + i];
    mat_elem_t hr1 = H[(r0 + 1) * n + i];
    mat_elem_t hr2 = H[(r0 + 2) * n + i];
    mat_elem_t dot = hr0 + v1 * hr1 + v2 * hr2;
    H[r0 * n + i] -= tau * dot;
    H[(r0 + 1) * n + i] -= tau * v1 * dot;
    H[(r0 + 2) * n + i] -= tau * v2 * dot;
  }

  // Accumulate Z = Z * P if requested
  if (Z) {
    for (size_t i = 0; i < n; i++) {
      mat_elem_t z0 = Z[r0 * ldz + i];
      mat_elem_t z1 = Z[(r0 + 1) * ldz + i];
      mat_elem_t z2 = Z[(r0 + 2) * ldz + i];
      mat_elem_t dot = z0 + v1 * z1 + v2 * z2;
      Z[r0 * ldz + i] -= tau * dot;
      Z[(r0 + 1) * ldz + i] -= tau * v1 * dot;
      Z[(r0 + 2) * ldz + i] -= tau * v2 * dot;
    }
  }
}

// Compute eigenvalues of a 2x2 block [[a, b], [c, d]]
// Returns real parts in sr[0], sr[1] and imaginary parts in si[0], si[1]
MAT_INTERNAL_STATIC void mat__eig_2x2_(mat_elem_t a, mat_elem_t b,
                                        mat_elem_t c, mat_elem_t d,
                                        mat_elem_t *sr, mat_elem_t *si) {
  mat_elem_t trace = a + d;
  mat_elem_t det = a * d - b * c;
  mat_elem_t disc = trace * trace - 4 * det;

  if (disc >= 0) {
    mat_elem_t sqrt_disc = MAT_SQRT(disc);
    sr[0] = (trace + sqrt_disc) / 2;
    sr[1] = (trace - sqrt_disc) / 2;
    si[0] = si[1] = 0;
  } else {
    sr[0] = sr[1] = trace / 2;
    si[0] = MAT_SQRT(-disc) / 2;
    si[1] = -si[0];
  }
}

// Extract shifts from the bottom-right nw x nw window of H[lo:hi+1, lo:hi+1]
// Returns number of shifts extracted (up to ns)
// Shifts are stored as (real, imag) pairs in sr[], si[]
MAT_INTERNAL_STATIC size_t mat__compute_shifts_(const Mat *H, size_t lo, size_t hi,
                                                 size_t ns, mat_elem_t *sr, mat_elem_t *si) {
  (void)lo;  // Unused but kept for API consistency
  size_t count = 0;

  // Extract eigenvalues from bottom-right 2x2 blocks
  size_t i = hi;
  while (count < ns && i >= lo + 1) {
    mat_elem_t sub = MAT_AT(H, i, i - 1);
    mat_elem_t diag_sum = MAT_FABS(MAT_AT(H, i - 1, i - 1)) + MAT_FABS(MAT_AT(H, i, i));

    if (MAT_FABS(sub) < MAT_DEFAULT_EPSILON * diag_sum || i == lo + 1) {
      // 1x1 block or small subdiagonal
      sr[count] = MAT_AT(H, i, i);
      si[count] = 0;
      count++;
      i--;
    } else {
      // 2x2 block
      mat__eig_2x2_(MAT_AT(H, i - 1, i - 1), MAT_AT(H, i - 1, i),
                   MAT_AT(H, i, i - 1), MAT_AT(H, i, i),
                   &sr[count], &si[count]);
      count += 2;
      i -= 2;
    }

    if (i < lo) break;
  }

  return count;
}

// Double-shift QR step using explicit shifts (for use with AED)
// Similar to mat_qr_step_ but uses provided shifts instead of computing from bottom 2x2
MAT_INTERNAL_STATIC void mat__qr_step_with_shifts_(Mat *H, size_t lo, size_t hi,
                                                    mat_elem_t s_re, mat_elem_t s_im,
                                                    mat_elem_t *Z, size_t ldz) {
  size_t n = H->rows;
  mat_elem_t *data = H->data;

  if (hi <= lo + 1) return;

  // For 2x2 block, use single shift (Wilkinson)
  if (hi - lo < 2) {
    mat__qr_step_(H, lo, hi, Z, ldz);
    return;
  }

  // Double shift using provided shifts (may be complex conjugate pair)
  // trace = 2 * s_re (sum of conjugate pair)
  // det = s_re^2 + s_im^2 (product of conjugate pair)
  mat_elem_t trace = 2 * s_re;
  mat_elem_t det = s_re * s_re + s_im * s_im;

  // First column of (H - s*I)(H - conj(s)*I) = H^2 - trace*H + det*I
  mat_elem_t h00 = MAT_AT(H, lo, lo);
  mat_elem_t h10 = MAT_AT(H, lo + 1, lo);
  mat_elem_t h01 = MAT_AT(H, lo, lo + 1);
  mat_elem_t h11 = MAT_AT(H, lo + 1, lo + 1);
  mat_elem_t h21 = MAT_AT(H, lo + 2, lo + 1);

  mat_elem_t x = h00 * h00 + h01 * h10 - trace * h00 + det;
  mat_elem_t y = h10 * (h00 + h11 - trace);
  mat_elem_t z = h10 * h21;

  // Chase bulge down using 3x3 Householder reflections
  for (size_t k = lo; k < hi - 1; k++) {
    mat_elem_t norm_sq = x * x + y * y + z * z;
    mat_elem_t norm = MAT_SQRT(norm_sq);

    if (norm < MAT_DEFAULT_EPSILON) {
      if (k < hi - 2) {
        x = MAT_AT(H, k + 1, k);
        y = MAT_AT(H, k + 2, k);
        z = (k + 3 <= hi) ? MAT_AT(H, k + 3, k) : 0;
      }
      continue;
    }

    mat_elem_t beta = (x >= 0) ? -norm : norm;
    mat_elem_t v0 = x - beta;
    mat_elem_t tau = -v0 / beta;
    mat_elem_t inv_v0 = 1.0 / v0;
    mat_elem_t v1 = y * inv_v0;
    mat_elem_t v2 = z * inv_v0;

    size_t r0 = k;

    // Apply Householder from left
    size_t col_start = (k > lo) ? k - 1 : lo;
    for (size_t j = col_start; j < n; j++) {
      mat_elem_t *col = &data[j * n];
      mat_elem_t dot = col[r0] + v1 * col[r0 + 1];
      if (r0 + 2 < n) dot += v2 * col[r0 + 2];
      col[r0] -= tau * dot;
      col[r0 + 1] -= tau * v1 * dot;
      if (r0 + 2 < n) col[r0 + 2] -= tau * v2 * dot;
    }

    // Apply Householder from right
    size_t row_end = (k + 4 < n) ? k + 4 : n;
    if (row_end > hi + 1) row_end = hi + 1;
    for (size_t i = 0; i < row_end; i++) {
      mat_elem_t hr0 = data[r0 * n + i];
      mat_elem_t hr1 = data[(r0 + 1) * n + i];
      mat_elem_t hr2 = (r0 + 2 < n) ? data[(r0 + 2) * n + i] : 0;
      mat_elem_t dot = hr0 + v1 * hr1 + v2 * hr2;
      data[r0 * n + i] -= tau * dot;
      data[(r0 + 1) * n + i] -= tau * v1 * dot;
      if (r0 + 2 < n) data[(r0 + 2) * n + i] -= tau * v2 * dot;
    }

    // Accumulate Z
    if (Z) {
      for (size_t i = 0; i < n; i++) {
        mat_elem_t z0 = Z[r0 * ldz + i];
        mat_elem_t z1 = Z[(r0 + 1) * ldz + i];
        mat_elem_t z2 = (r0 + 2 < n) ? Z[(r0 + 2) * ldz + i] : 0;
        mat_elem_t dot = z0 + v1 * z1 + v2 * z2;
        Z[r0 * ldz + i] -= tau * dot;
        Z[(r0 + 1) * ldz + i] -= tau * v1 * dot;
        if (r0 + 2 < n) Z[(r0 + 2) * ldz + i] -= tau * v2 * dot;
      }
    }

    // Pick up bulge for next iteration
    if (k < hi - 2) {
      x = MAT_AT(H, k + 1, k);
      y = MAT_AT(H, k + 2, k);
      z = (k + 3 <= hi) ? MAT_AT(H, k + 3, k) : 0;
    }
  }

  // Final 2x2 Givens rotation
  mat_elem_t x_final = MAT_AT(H, hi - 1, hi - 2);
  mat_elem_t y_final = MAT_AT(H, hi, hi - 2);
  mat_elem_t c_final, s_final;
  mat__givens_(x_final, y_final, &c_final, &s_final);

  for (size_t j = hi - 2; j < n; j++) {
    mat_elem_t *col = &data[j * n];
    mat_elem_t t0 = col[hi - 1];
    mat_elem_t t1 = col[hi];
    col[hi - 1] = c_final * t0 + s_final * t1;
    col[hi] = -s_final * t0 + c_final * t1;
  }
  mat__givens_cols_(data, n, hi + 1, hi - 1, hi, c_final, s_final);

  if (Z) {
    mat_elem_t *Z_hi1 = &Z[(hi - 1) * ldz];
    mat_elem_t *Z_hi = &Z[hi * ldz];
    for (size_t i = 0; i < n; i++) {
      mat_elem_t z0 = Z_hi1[i];
      mat_elem_t z1 = Z_hi[i];
      Z_hi1[i] = c_final * z0 + s_final * z1;
      Z_hi[i] = -s_final * z0 + c_final * z1;
    }
  }
}

// Deflation tolerance: use max of relative and absolute threshold
// Absolute floor prevents issues when diagonals are near-zero
// Must be large enough to detect "stuck" subdiagonals from numerical noise
// For single precision, values < 1e-5 are effectively zero after many operations
#ifdef MAT_DOUBLE_PRECISION
#define MAT_DEFLATION_ABS_TOL 1e-12
#else
#define MAT_DEFLATION_ABS_TOL 2e-6f
#endif

MAT_INTERNAL_STATIC int mat__is_negligible_(mat_elem_t sub, mat_elem_t diag_sum) {
  mat_elem_t rel_tol = MAT_DEFAULT_EPSILON * diag_sum;
  mat_elem_t tol = (rel_tol > MAT_DEFLATION_ABS_TOL) ? rel_tol : MAT_DEFLATION_ABS_TOL;
  return MAT_FABS(sub) < tol;
}

// Check for deflation in subdiagonal
// Returns new hi after deflating converged eigenvalues
MAT_INTERNAL_STATIC size_t mat__check_deflation_(Mat *H, size_t lo, size_t hi) {
  while (hi > lo) {
    mat_elem_t sub = MAT_AT(H, hi, hi - 1);
    mat_elem_t diag_sum = MAT_FABS(MAT_AT(H, hi - 1, hi - 1)) + MAT_FABS(MAT_AT(H, hi, hi));
    if (mat__is_negligible_(sub, diag_sum)) {
      MAT_SET(H, hi, hi - 1, 0);
      hi--;
    } else {
      // Check for 2x2 block
      if (hi > lo + 1) {
        mat_elem_t sub2 = MAT_AT(H, hi - 1, hi - 2);
        mat_elem_t diag_sum2 = MAT_FABS(MAT_AT(H, hi - 2, hi - 2)) + MAT_FABS(MAT_AT(H, hi - 1, hi - 1));
        if (mat__is_negligible_(sub2, diag_sum2)) {
          MAT_SET(H, hi - 1, hi - 2, 0);
          hi -= 2;
        } else {
          break;
        }
      } else {
        break;
      }
    }
  }
  return hi;
}

// Aggressive Early Deflation (AED)
// Examines a deflation window at the bottom-right and deflates converged eigenvalues
// Returns the number of eigenvalues deflated
MAT_INTERNAL_STATIC size_t mat__aed_(Mat *H, size_t lo, size_t hi,
                                      size_t nw, mat_elem_t *sr, mat_elem_t *si,
                                      mat_elem_t *Z, size_t ldz) {
  size_t n = H->rows;
  if (hi - lo + 1 < nw) nw = hi - lo + 1;
  if (nw < 3) return 0;

  size_t kwtop = hi - nw + 1;  // Top of deflation window

  // Copy the deflation window to a temporary matrix
  Mat *T = mat_mat(nw, nw);
  for (size_t j = 0; j < nw; j++) {
    for (size_t i = 0; i < nw; i++) {
      MAT_SET(T, i, j, MAT_AT(H, kwtop + i, kwtop + j));
    }
  }

  // Initialize Q for the window
  Mat *Q = mat_mat(nw, nw);
  mat_eye(Q);

  // Reduce the window to quasi-triangular form using Francis QR
  size_t w_hi = nw - 1;
  size_t max_iter = 30 * nw;
  for (size_t iter = 0; iter < max_iter && w_hi > 0; iter++) {
    size_t w_lo = w_hi;
    while (w_lo > 0) {
      mat_elem_t sub = MAT_AT(T, w_lo, w_lo - 1);
      mat_elem_t diag_sum = MAT_FABS(MAT_AT(T, w_lo - 1, w_lo - 1)) +
                            MAT_FABS(MAT_AT(T, w_lo, w_lo));
      if (MAT_FABS(sub) < MAT_DEFAULT_EPSILON * diag_sum) {
        MAT_SET(T, w_lo, w_lo - 1, 0);
        break;
      }
      w_lo--;
    }
    if (w_lo == w_hi) {
      w_hi--;
    } else if (w_lo + 1 == w_hi) {
      w_hi -= 2;
    } else {
      mat__qr_step_(T, w_lo, w_hi, Q->data, nw);
    }
  }

  // Count deflatable eigenvalues
  // An eigenvalue is deflatable if the "spike" (coupling to rest of matrix) is small
  mat_elem_t spike = MAT_AT(H, kwtop, kwtop - 1);  // The coupling element
  size_t ns = 0;  // Number of shifts (non-deflated eigenvalues)
  size_t nd = 0;  // Number of deflated eigenvalues

  size_t i = nw - 1;
  while (i < nw) {  // Unsigned underflow check
    // Check if this is a 2x2 block
    int is_2x2 = 0;
    if (i > 0) {
      mat_elem_t sub = MAT_AT(T, i, i - 1);
      mat_elem_t diag_sum = MAT_FABS(MAT_AT(T, i - 1, i - 1)) + MAT_FABS(MAT_AT(T, i, i));
      if (MAT_FABS(sub) >= MAT_DEFAULT_EPSILON * diag_sum) {
        is_2x2 = 1;
      }
    }

    if (is_2x2) {
      // Check coupling for 2x2 block
      mat_elem_t coupling = MAT_FABS(spike) * (MAT_FABS(Q->data[(i - 1) * nw]) +
                                                MAT_FABS(Q->data[i * nw]));
      mat_elem_t test = MAT_FABS(MAT_AT(T, i - 1, i - 1)) + MAT_FABS(MAT_AT(T, i, i));
      if (coupling < MAT_DEFAULT_EPSILON * test) {
        // Deflatable
        nd += 2;
      } else {
        // Not deflatable - use as shifts
        mat__eig_2x2_(MAT_AT(T, i - 1, i - 1), MAT_AT(T, i - 1, i),
                     MAT_AT(T, i, i - 1), MAT_AT(T, i, i),
                     &sr[ns], &si[ns]);
        ns += 2;
      }
      if (i < 2) break;
      i -= 2;
    } else {
      // 1x1 block
      mat_elem_t coupling = MAT_FABS(spike) * MAT_FABS(Q->data[i * nw]);
      mat_elem_t test = MAT_FABS(MAT_AT(T, i, i));
      if (coupling < MAT_DEFAULT_EPSILON * test) {
        // Deflatable
        nd++;
      } else {
        // Not deflatable - use as shift
        sr[ns] = MAT_AT(T, i, i);
        si[ns] = 0;
        ns++;
      }
      if (i == 0) break;
      i--;
    }
  }

  // AED deflation is only supported when Z is provided, because the current
  // implementation doesn't properly reorder eigenvalues. Without Z, we just
  // use the computed shifts and return 0 deflations.
  if (!Z) {
    MAT_FREE_MAT(T);
    MAT_FREE_MAT(Q);
    return 0;  // No deflation, but sr/si contains shifts to use
  }

  // If we deflated eigenvalues, apply Q back to the original matrix
  if (nd > 0) {
    // Allocate temp buffer
    mat_elem_t *temp = (mat_elem_t *)MAT_MALLOC(nw * sizeof(mat_elem_t));

    // H[kwtop:hi+1, kwtop:hi+1] = T (already has Q applied via Schur decomposition)
    for (size_t j = 0; j < nw; j++) {
      for (size_t ii = 0; ii < nw; ii++) {
        MAT_SET(H, kwtop + ii, kwtop + j, MAT_AT(T, ii, j));
      }
    }

    // Update the spike: H[kwtop, kwtop-1] = spike * Q[0, 0]
    if (kwtop > lo) {
      mat_elem_t new_spike = spike * Q->data[0];
      MAT_SET(H, kwtop, kwtop - 1, new_spike);
    }

    // Apply Q from right to H[0:kwtop, kwtop:hi+1]
    for (size_t ii = 0; ii < kwtop; ii++) {
      for (size_t j = 0; j < nw; j++) {
        temp[j] = 0;
        for (size_t k = 0; k < nw; k++) {
          temp[j] += MAT_AT(H, ii, kwtop + k) * MAT_AT(Q, k, j);
        }
      }
      for (size_t j = 0; j < nw; j++) {
        MAT_SET(H, ii, kwtop + j, temp[j]);
      }
    }

    // Apply Q from left to H[kwtop:hi+1, hi+1:n]
    for (size_t j = hi + 1; j < n; j++) {
      for (size_t ii = 0; ii < nw; ii++) {
        temp[ii] = 0;
        for (size_t k = 0; k < nw; k++) {
          temp[ii] += MAT_AT(Q, k, ii) * MAT_AT(H, kwtop + k, j);
        }
      }
      for (size_t ii = 0; ii < nw; ii++) {
        MAT_SET(H, kwtop + ii, j, temp[ii]);
      }
    }

    // Accumulate Q into Z: Z[:, kwtop:hi+1] = Z[:, kwtop:hi+1] * Q
    for (size_t ii = 0; ii < n; ii++) {
      for (size_t j = 0; j < nw; j++) {
        temp[j] = 0;
        for (size_t k = 0; k < nw; k++) {
          temp[j] += Z[(kwtop + k) * ldz + ii] * MAT_AT(Q, k, j);
        }
      }
      for (size_t j = 0; j < nw; j++) {
        Z[(kwtop + j) * ldz + ii] = temp[j];
      }
    }

    MAT_FREE(temp);
  }

  MAT_FREE_MAT(T);
  MAT_FREE_MAT(Q);

  return nd;
}

// Main multishift QR iteration with optional AED
// H is reduced to quasi-upper-triangular form
// If Z is provided, accumulate transformations
MAT_INTERNAL_STATIC void mat__multishift_qr_iter_(Mat *H, mat_elem_t *Z, size_t ldz) {
  size_t n = H->rows;
  if (n <= 2) return;

  size_t max_iter = 30 * n;
  size_t hi = n - 1;
  size_t iter = 0;

  while (hi > 0 && iter < max_iter) {
    // Check for deflation from the bottom
    hi = mat__check_deflation_(H, 0, hi);
    if (hi == 0) break;

    // Find the start of the active block
    size_t lo = hi;
    while (lo > 0) {
      mat_elem_t sub = MAT_AT(H, lo, lo - 1);
      mat_elem_t diag_sum = MAT_FABS(MAT_AT(H, lo - 1, lo - 1)) + MAT_FABS(MAT_AT(H, lo, lo));
      if (mat__is_negligible_(sub, diag_sum)) {
        MAT_SET(H, lo, lo - 1, 0);
        break;
      }
      lo--;
    }

    size_t block_size = hi - lo + 1;

    if (block_size == 1) {
      // Single eigenvalue converged
      hi--;
    } else if (block_size == 2) {
      // 2x2 block converged
      if (hi < 2) break;  // Guard against underflow
      hi -= 2;
    } else {
      mat__qr_step_(H, lo, hi, Z, ldz);
      iter++;
    }
  }
}

// Extract eigenvalues from upper quasi-triangular matrix
// Returns number of eigenvalues found
MAT_INTERNAL_STATIC size_t mat__extract_eigvals_(Vec *out, const Mat *H,
                                                  size_t lo, size_t hi) {
  size_t count = 0;

  for (size_t i = lo; i <= hi;) {
    if (i == hi) {
      // 1x1 block: real eigenvalue
      out->data[count++] = MAT_AT(H, i, i);
      i++;
    } else {
      mat_elem_t sub = MAT_AT(H, i + 1, i);
      if (MAT_FABS(sub) < MAT_DEFAULT_EPSILON *
              (MAT_FABS(MAT_AT(H, i, i)) + MAT_FABS(MAT_AT(H, i + 1, i + 1)))) {
        // Negligible subdiagonal: treat as 1x1 block
        out->data[count++] = MAT_AT(H, i, i);
        i++;
      } else {
        // 2x2 block: may have complex eigenvalues
        mat_elem_t a = MAT_AT(H, i, i);
        mat_elem_t b = MAT_AT(H, i, i + 1);
        mat_elem_t c = MAT_AT(H, i + 1, i);
        mat_elem_t d = MAT_AT(H, i + 1, i + 1);

        mat_elem_t trace = a + d;
        mat_elem_t det = a * d - b * c;
        mat_elem_t disc = trace * trace - 4 * det;

        if (disc >= 0) {
          // Real eigenvalues
          mat_elem_t sqrt_disc = MAT_SQRT(disc);
          out->data[count++] = (trace + sqrt_disc) / 2;
          out->data[count++] = (trace - sqrt_disc) / 2;
        } else {
          // Complex conjugate pair: store real parts
          out->data[count++] = trace / 2;
          out->data[count++] = trace / 2;
        }
        i += 2;
      }
    }
  }

  return count;
}

// Compute eigenvectors from quasi-upper-triangular Schur form T.
// T is n x n, V is n x n output (eigenvectors as columns).
// For 1x1 blocks (real eigenvalue): solve (T - λI) * y = 0 by back-substitution.
// For 2x2 blocks (complex pair): compute real/imag parts of eigenvector.
// Output: V[:, k] = eigenvector for eigenvalue at position k.
//         For complex pairs at k, k+1: V[:, k] = real part, V[:, k+1] = imag part.
MAT_INTERNAL_STATIC void mat__trevc_(Mat *V, const Mat *T) {
  size_t n = T->rows;
  mat_elem_t *Vd = V->data;

  // Initialize V to zero
  memset(Vd, 0, n * n * sizeof(mat_elem_t));

  // Work through diagonal blocks
  for (size_t k = 0; k < n;) {
    // Check if this is a 2x2 block
    int is_2x2 = 0;
    if (k + 1 < n) {
      mat_elem_t sub = MAT_AT(T, k + 1, k);
      mat_elem_t diag_sum = MAT_FABS(MAT_AT(T, k, k)) + MAT_FABS(MAT_AT(T, k + 1, k + 1));
      if (MAT_FABS(sub) >= MAT_DEFAULT_EPSILON * diag_sum) {
        is_2x2 = 1;
      }
    }

    if (!is_2x2) {
      // 1x1 block: real eigenvalue λ = T[k,k]
      mat_elem_t lambda = MAT_AT(T, k, k);
      mat_elem_t *y = &Vd[k * n];  // Column k of V

      // Set y[k] = 1
      y[k] = 1.0;

      // Back-solve: for j = k-1 down to 0
      // y[j] = -sum(T[j,l] * y[l] for l=j+1..k) / (T[j,j] - λ)
      for (size_t j_idx = 0; j_idx < k; j_idx++) {
        size_t j = k - 1 - j_idx;  // j = k-1, k-2, ..., 0
        mat_elem_t sum = 0;
        for (size_t l = j + 1; l <= k; l++) {
          sum += MAT_AT(T, j, l) * y[l];
        }
        mat_elem_t denom = MAT_AT(T, j, j) - lambda;
        if (MAT_FABS(denom) < MAT_DEFAULT_EPSILON) {
          denom = MAT_DEFAULT_EPSILON;  // Avoid division by zero
        }
        y[j] = -sum / denom;
      }

      // Normalize
      mat_elem_t norm = 0;
      for (size_t i = 0; i <= k; i++) {
        norm += y[i] * y[i];
      }
      norm = MAT_SQRT(norm);
      if (norm > MAT_DEFAULT_EPSILON) {
        for (size_t i = 0; i <= k; i++) {
          y[i] /= norm;
        }
      }

      k++;
    } else {
      // 2x2 block: complex conjugate eigenvalues
      // [[a, b], [c, d]] with eigenvalues (a+d)/2 ± i*sqrt(bc - ((a-d)/2)^2)
      mat_elem_t a = MAT_AT(T, k, k);
      mat_elem_t b = MAT_AT(T, k, k + 1);
      mat_elem_t c = MAT_AT(T, k + 1, k);
      mat_elem_t d = MAT_AT(T, k + 1, k + 1);

      mat_elem_t re = (a + d) / 2;  // Real part of eigenvalue
      mat_elem_t disc = (a - d) * (a - d) / 4 + b * c;
      mat_elem_t im = MAT_SQRT(MAT_FABS(-disc));  // Imaginary part

      // For complex eigenvalue λ = re + i*im, eigenvector [p + i*q] satisfies:
      // (T - λI) * (p + i*q) = 0
      // This gives us two real equations we can solve.
      // For the 2x2 block, eigenvector is [1, (λ-a)/b] or [-(λ-d)/c, 1]
      // With λ = re + i*im, we get (p, q) for real and imaginary parts.

      mat_elem_t *y_re = &Vd[k * n];        // Real part in column k
      mat_elem_t *y_im = &Vd[(k + 1) * n];  // Imag part in column k+1

      // Simple eigenvector for 2x2: take [1, (re-a)/b + i*(im/b)]
      if (MAT_FABS(b) > MAT_DEFAULT_EPSILON) {
        y_re[k] = 1.0;
        y_im[k] = 0.0;
        y_re[k + 1] = (re - a) / b;
        y_im[k + 1] = im / b;
      } else if (MAT_FABS(c) > MAT_DEFAULT_EPSILON) {
        y_re[k] = (re - d) / (-c);
        y_im[k] = im / (-c);
        y_re[k + 1] = 1.0;
        y_im[k + 1] = 0.0;
      } else {
        // Degenerate case
        y_re[k] = 1.0;
        y_im[k] = 0.0;
        y_re[k + 1] = 0.0;
        y_im[k + 1] = 1.0;
      }

      // Back-solve for the rest of the matrix
      // Solve (T[0:k, 0:k] - re*I) * p - im * q = -T[0:k, k:k+2] * [p_k; p_{k+1}]
      // and   im * p + (T[0:k, 0:k] - re*I) * q = -T[0:k, k:k+2] * [q_k; q_{k+1}]
      for (size_t j_idx = 0; j_idx < k; j_idx++) {
        size_t j = k - 1 - j_idx;

        // RHS from known part (rows k and k+1)
        mat_elem_t rhs_re = -(MAT_AT(T, j, k) * y_re[k] + MAT_AT(T, j, k + 1) * y_re[k + 1]);
        mat_elem_t rhs_im = -(MAT_AT(T, j, k) * y_im[k] + MAT_AT(T, j, k + 1) * y_im[k + 1]);

        // Add contribution from already computed components (j+1 to k-1)
        for (size_t l = j + 1; l < k; l++) {
          rhs_re -= MAT_AT(T, j, l) * y_re[l];
          rhs_im -= MAT_AT(T, j, l) * y_im[l];
        }

        // Solve 2x2 system:
        // (T[j,j] - re) * p[j] - im * q[j] = rhs_re
        // im * p[j] + (T[j,j] - re) * q[j] = rhs_im
        mat_elem_t tjj = MAT_AT(T, j, j);
        mat_elem_t alpha = tjj - re;
        mat_elem_t det = alpha * alpha + im * im;
        if (MAT_FABS(det) < MAT_DEFAULT_EPSILON * MAT_DEFAULT_EPSILON) {
          det = MAT_DEFAULT_EPSILON * MAT_DEFAULT_EPSILON;
        }
        y_re[j] = (alpha * rhs_re + im * rhs_im) / det;
        y_im[j] = (alpha * rhs_im - im * rhs_re) / det;
      }

      // Normalize both vectors
      mat_elem_t norm_re = 0, norm_im = 0;
      for (size_t i = 0; i <= k + 1; i++) {
        norm_re += y_re[i] * y_re[i];
        norm_im += y_im[i] * y_im[i];
      }
      norm_re = MAT_SQRT(norm_re);
      norm_im = MAT_SQRT(norm_im);
      if (norm_re > MAT_DEFAULT_EPSILON) {
        for (size_t i = 0; i <= k + 1; i++) {
          y_re[i] /= norm_re;
        }
      }
      if (norm_im > MAT_DEFAULT_EPSILON) {
        for (size_t i = 0; i <= k + 1; i++) {
          y_im[i] /= norm_im;
        }
      }

      k += 2;
    }
  }
}

/* Symmetric Eigenvalues */

// Block size for blocked tridiagonalization
#ifndef MAT_TRIDIAG_BLOCK_SIZE
#define MAT_TRIDIAG_BLOCK_SIZE 64
#endif

// Threshold for using blocked algorithm (smaller matrices use unblocked)
#ifndef MAT_TRIDIAG_BLOCK_THRESHOLD
#define MAT_TRIDIAG_BLOCK_THRESHOLD 128
#endif

// Unblocked tridiagonalization - used for small matrices and within blocks
// If Q != NULL, accumulates Householder reflectors into Q (must be initialized to identity)
MAT_INTERNAL_STATIC void mat__tridiag_sym_unblocked_(mat_elem_t *W, size_t n,
                                                      mat_elem_t *d, mat_elem_t *e,
                                                      mat_elem_t *v, mat_elem_t *p,
                                                      mat_elem_t *Q, size_t ldq) {
  mat_elem_t *w = NULL;
  if (Q) {
    w = (mat_elem_t *)MAT_MALLOC(n * sizeof(mat_elem_t));
  }

  for (size_t k = 0; k < n - 1; k++) {
    size_t len = n - k - 1;
    mat_elem_t *col_k = &W[k * n + (k + 1)];

    memcpy(v, col_k, len * sizeof(mat_elem_t));

    mat_elem_t norm_sq = mat__dot_kernel(v, v, len);
    mat_elem_t norm_v = MAT_SQRT(norm_sq);

    if (norm_v < MAT_DEFAULT_EPSILON) {
      d[k] = W[k * n + k];
      e[k] = 0;
      continue;
    }

    mat_elem_t v0 = v[0];
    mat_elem_t beta = (v0 >= 0) ? -norm_v : norm_v;
    e[k] = -beta;

    v[0] = v0 - beta;
    mat_elem_t inv_v0 = 1.0 / v[0];
    mat__scal_kernel(&v[1], inv_v0, len - 1);
    v[0] = 1.0;

    mat_elem_t tau = (beta - v0) / beta;

    col_k[0] = beta;
    memset(&col_k[1], 0, (len - 1) * sizeof(mat_elem_t));
    W[(k + 1) * n + k] = beta;
    for (size_t i = 2; i <= len; i++) {
      W[(k + i) * n + k] = 0;
    }

    memset(p, 0, len * sizeof(mat_elem_t));
    for (size_t j = 0; j < len; j++) {
      mat_elem_t *col_j = &W[(k + 1 + j) * n + (k + 1)];
      mat__axpy_kernel(p, v[j], col_j, len);
    }

    mat_elem_t vtp = mat__dot_kernel(v, p, len);
    mat_elem_t alpha = (tau * tau / 2) * vtp;

    mat__scal_kernel(p, tau, len);
    mat__axpy_kernel(p, -alpha, v, len);

    for (size_t j = 0; j < len; j++) {
      mat_elem_t *col_j = &W[(k + 1 + j) * n + (k + 1)];
      mat__syr2_col_kernel(col_j, v[j], p, p[j], v, len);
    }

    // Accumulate Q: Q = Q * H = Q - tau * (Q * v) * v^T
    if (Q) {
      memset(w, 0, n * sizeof(mat_elem_t));
      for (size_t j = 0; j < len; j++) {
        mat_elem_t *Q_col = &Q[(k + 1 + j) * ldq];
        mat__axpy_kernel(w, v[j], Q_col, n);
      }
      for (size_t j = 0; j < len; j++) {
        mat_elem_t *Q_col = &Q[(k + 1 + j) * ldq];
        mat__axpy_kernel(Q_col, -tau * v[j], w, n);
      }
    }

    d[k] = W[k * n + k];
  }

  d[n - 1] = W[(n - 1) * n + (n - 1)];

  if (w) MAT_FREE(w);
}

// Blocked tridiagonalization using SYR2K for trailing matrix update
// Reduces nb columns at a time, accumulates V and W, then does rank-2*nb update
MAT_INTERNAL_STATIC void mat__tridiag_sym_blocked_(mat_elem_t *A, size_t n,
                                                    mat_elem_t *d, mat_elem_t *e,
                                                    size_t nb) {
  mat_elem_t *v = (mat_elem_t *)MAT_MALLOC(n * sizeof(mat_elem_t));
  mat_elem_t *p = (mat_elem_t *)MAT_MALLOC(n * sizeof(mat_elem_t));
  mat_elem_t *V = (mat_elem_t *)MAT_MALLOC(n * nb * sizeof(mat_elem_t));
  mat_elem_t *W_mat = (mat_elem_t *)MAT_MALLOC(n * nb * sizeof(mat_elem_t));
  mat_elem_t *tau_arr = (mat_elem_t *)MAT_MALLOC(nb * sizeof(mat_elem_t));

  for (size_t k = 0; k < n - 1; k += nb) {
    size_t kb = (k + nb < n - 1) ? nb : (n - 1 - k);  // actual block size
    size_t trail_start = k + kb + 1;
    size_t trail_len = (trail_start < n) ? n - trail_start : 0;

    // Panel factorization: reduce columns k to k+kb-1
    for (size_t j = 0; j < kb; j++) {
      size_t col_idx = k + j;
      size_t len = n - col_idx - 1;
      mat_elem_t *col = &A[col_idx * n + (col_idx + 1)];

      memcpy(v, col, len * sizeof(mat_elem_t));

      mat_elem_t norm_sq = mat__dot_kernel(v, v, len);
      mat_elem_t norm_v = MAT_SQRT(norm_sq);

      if (norm_v < MAT_DEFAULT_EPSILON) {
        d[col_idx] = A[col_idx * n + col_idx];
        e[col_idx] = 0;
        tau_arr[j] = 0;
        memset(&V[j * n], 0, n * sizeof(mat_elem_t));
        memset(&W_mat[j * n], 0, n * sizeof(mat_elem_t));
        continue;
      }

      mat_elem_t v0 = v[0];
      mat_elem_t beta = (v0 >= 0) ? -norm_v : norm_v;
      e[col_idx] = -beta;

      v[0] = v0 - beta;
      mat_elem_t inv_v0 = 1.0 / v[0];
      mat__scal_kernel(&v[1], inv_v0, len - 1);
      v[0] = 1.0;

      mat_elem_t tau = (beta - v0) / beta;
      tau_arr[j] = tau;

      // Store v in V (padded with zeros for rows < col_idx+1)
      memset(&V[j * n], 0, (col_idx + 1) * sizeof(mat_elem_t));
      memcpy(&V[j * n + col_idx + 1], v, len * sizeof(mat_elem_t));

      col[0] = beta;
      memset(&col[1], 0, (len - 1) * sizeof(mat_elem_t));
      A[(col_idx + 1) * n + col_idx] = beta;
      for (size_t i = 2; i <= len; i++) {
        A[(col_idx + i) * n + col_idx] = 0;
      }

      // Compute p = A * v for the remaining panel
      memset(p, 0, len * sizeof(mat_elem_t));
      for (size_t jj = 0; jj < len; jj++) {
        mat_elem_t *col_jj = &A[(col_idx + 1 + jj) * n + (col_idx + 1)];
        mat__axpy_kernel(p, v[jj], col_jj, len);
      }

      mat_elem_t vtp = mat__dot_kernel(v, p, len);
      mat_elem_t alpha = (tau * tau / 2) * vtp;

      mat__scal_kernel(p, tau, len);
      mat__axpy_kernel(p, -alpha, v, len);

      // Store w in W_mat (padded)
      memset(&W_mat[j * n], 0, (col_idx + 1) * sizeof(mat_elem_t));
      memcpy(&W_mat[j * n + col_idx + 1], p, len * sizeof(mat_elem_t));

      // Update only the remaining panel columns (within-panel update)
      for (size_t jj = j + 1; jj < kb; jj++) {
        size_t cjj = k + jj;
        if (cjj >= n - 1) break;
        mat_elem_t *col_jj = &A[cjj * n + (col_idx + 1)];
        size_t upd_len = n - col_idx - 1;
        if (cjj >= col_idx + 1) {
          size_t off = cjj - col_idx - 1;
          mat__syr2_col_kernel(&col_jj[0], v[off], &p[0], p[off], &v[0], upd_len);
        }
      }

      d[col_idx] = A[col_idx * n + col_idx];
    }

    // Trailing matrix update: A[trail:n, trail:n] -= V*W' + W*V' using SYR2K
    if (trail_len > 1) {
      Mat C_sub = {.data = &A[trail_start * n + trail_start],
                   .rows = trail_len, .cols = trail_len};

      // V and W_mat are stored with stride n, need contiguous copies for SYR2K
      mat_elem_t *V_trail = (mat_elem_t *)MAT_MALLOC(trail_len * kb * sizeof(mat_elem_t));
      mat_elem_t *W_trail = (mat_elem_t *)MAT_MALLOC(trail_len * kb * sizeof(mat_elem_t));

      for (size_t j = 0; j < kb; j++) {
        memcpy(&V_trail[j * trail_len], &V[j * n + trail_start],
               trail_len * sizeof(mat_elem_t));
        memcpy(&W_trail[j * trail_len], &W_mat[j * n + trail_start],
               trail_len * sizeof(mat_elem_t));
      }

      Mat V_t = {.data = V_trail, .rows = trail_len, .cols = kb};
      Mat W_t = {.data = W_trail, .rows = trail_len, .cols = kb};

      mat_syr2k(&C_sub, &V_t, &W_t, -1.0, 1.0, 'L');

      // Copy lower to upper for symmetry
      for (size_t i = 0; i < trail_len; i++) {
        for (size_t jj = i + 1; jj < trail_len; jj++) {
          A[(trail_start + jj) * n + trail_start + i] =
              A[(trail_start + i) * n + trail_start + jj];
        }
      }

      MAT_FREE(W_trail);
      MAT_FREE(V_trail);
    }
  }

  MAT_FREE(tau_arr);
  MAT_FREE(W_mat);
  MAT_FREE(V);
  MAT_FREE(p);
  MAT_FREE(v);
}

// Reduce symmetric matrix to tridiagonal form using Householder reflections
// Output: diagonal d[n], off-diagonal e[n-1]
// Uses blocked algorithm for large matrices, unblocked for small
MAT_INTERNAL_STATIC void mat__tridiag_sym_(const Mat *A, mat_elem_t *d, mat_elem_t *e) {
  size_t n = A->rows;

  mat_elem_t *W = (mat_elem_t *)MAT_MALLOC(n * n * sizeof(mat_elem_t));
  memcpy(W, A->data, n * n * sizeof(mat_elem_t));

  if (n >= MAT_TRIDIAG_BLOCK_THRESHOLD) {
    mat__tridiag_sym_blocked_(W, n, d, e, MAT_TRIDIAG_BLOCK_SIZE);
  } else {
    mat_elem_t *v = (mat_elem_t *)MAT_MALLOC(n * sizeof(mat_elem_t));
    mat_elem_t *p = (mat_elem_t *)MAT_MALLOC(n * sizeof(mat_elem_t));
    mat__tridiag_sym_unblocked_(W, n, d, e, v, p, NULL, 0);
    MAT_FREE(p);
    MAT_FREE(v);
  }

  MAT_FREE(W);
}

// Single implicit QR step with Wilkinson shift on tridiagonal matrix
// Updates diagonal d[lo:hi+1] and off-diagonal e[lo:hi]
// If Z != NULL, accumulates Givens rotations into Z for eigenvector computation
// Reference: Golub & Van Loan, "Matrix Computations", Algorithm 8.3.2
MAT_INTERNAL_STATIC void mat__tridiag_qr_step_(mat_elem_t *d, mat_elem_t *e,
                                                size_t lo, size_t hi,
                                                mat_elem_t *Z, size_t n, size_t ldz) {
  // Wilkinson shift from bottom-right 2x2 block [[a,b],[b,c]]
  mat_elem_t a = d[hi - 1];
  mat_elem_t b = e[hi - 1];
  mat_elem_t cc = d[hi];

  mat_elem_t delta = (a - cc) / 2;
  mat_elem_t sign_delta = (delta >= 0) ? 1 : -1;
  mat_elem_t shift = cc - sign_delta * b * b /
                     (MAT_FABS(delta) + MAT_SQRT(delta * delta + b * b));

  // Initialize bulge chase
  mat_elem_t f = d[lo] - shift;
  mat_elem_t g = e[lo];

  for (size_t k = lo; k < hi; k++) {
    // Compute Givens rotation to zero g
    mat_elem_t cs, sn, r;
    if (g == 0) {
      cs = 1;
      sn = 0;
      r = f;
    } else if (MAT_FABS(g) > MAT_FABS(f)) {
      mat_elem_t t = f / g;
      mat_elem_t tt = MAT_SQRT(1 + t * t);
      sn = 1 / tt;
      cs = sn * t;
      r = g * tt;
    } else {
      mat_elem_t t = g / f;
      mat_elem_t tt = MAT_SQRT(1 + t * t);
      cs = 1 / tt;
      sn = cs * t;
      r = f * tt;
    }

    // Update off-diagonal from previous iteration
    if (k > lo) {
      e[k - 1] = r;
    }

    // Apply symmetric Givens rotation to 2x2 block [[d[k], e[k]], [e[k], d[k+1]]]
    mat_elem_t dk = d[k];
    mat_elem_t ek = e[k];
    mat_elem_t dk1 = d[k + 1];

    d[k] = cs * cs * dk + 2 * cs * sn * ek + sn * sn * dk1;
    d[k + 1] = sn * sn * dk - 2 * cs * sn * ek + cs * cs * dk1;
    e[k] = cs * sn * (dk1 - dk) + (cs * cs - sn * sn) * ek;

    // Create bulge for next iteration
    if (k < hi - 1) {
      f = e[k];
      g = sn * e[k + 1];
      e[k + 1] = cs * e[k + 1];
    }

    // Accumulate Givens into Z: Z = Z * G where G = [[cs, sn], [-sn, cs]]
    if (Z) {
      mat__givens_cols_(Z, ldz, n, k, k + 1, cs, -sn);
    }
  }
}

/* Tridiagonal QR Iteration */

// Full QR iteration on tridiagonal matrix with deflation
// If Z != NULL, accumulates Givens rotations into Z for eigenvector computation
MAT_INTERNAL_STATIC void mat__tridiag_qr_iter_(mat_elem_t *d, mat_elem_t *e, size_t n,
                                                mat_elem_t *Z, size_t ldz) {
  if (n <= 1) return;

  size_t max_iter = 30 * n;
  size_t hi = n - 1;

  for (size_t iter = 0; iter < max_iter && hi > 0; iter++) {
    // Find active submatrix (deflate converged eigenvalues)
    size_t lo = hi;
    while (lo > 0) {
      mat_elem_t off = MAT_FABS(e[lo - 1]);
      mat_elem_t diag_sum = MAT_FABS(d[lo - 1]) + MAT_FABS(d[lo]);
      if (off < MAT_DEFAULT_EPSILON * diag_sum) {
        e[lo - 1] = 0;
        break;
      }
      lo--;
    }

    if (lo == hi) {
      // 1x1 block converged
      hi--;
    } else {
      // Perform QR step on active submatrix
      mat__tridiag_qr_step_(d, e, lo, hi, Z, n, ldz);
    }
  }
}

// Scalar implementation of symmetric eigenvalue computation
MAT_INTERNAL_STATIC void mat__eigvals_sym_scalar_(Vec *out, const Mat *A) {
  size_t n = A->rows;

  // Handle small cases directly
  if (n == 1) {
    out->data[0] = MAT_AT(A, 0, 0);
    return;
  }

  if (n == 2) {
    mat_elem_t a = MAT_AT(A, 0, 0);
    mat_elem_t b = MAT_AT(A, 0, 1);
    mat_elem_t d = MAT_AT(A, 1, 1);

    mat_elem_t trace = a + d;
    mat_elem_t det = a * d - b * b;  // symmetric: c = b
    mat_elem_t disc = trace * trace - 4 * det;

    mat_elem_t sqrt_disc = (disc > 0) ? MAT_SQRT(disc) : 0;
    out->data[0] = (trace + sqrt_disc) / 2;
    out->data[1] = (trace - sqrt_disc) / 2;
    return;
  }

  // Allocate diagonal and off-diagonal arrays
  mat_elem_t *diag = (mat_elem_t *)MAT_MALLOC(n * sizeof(mat_elem_t));
  mat_elem_t *offdiag = (mat_elem_t *)MAT_MALLOC((n - 1) * sizeof(mat_elem_t));

  // Reduce to tridiagonal form
  mat__tridiag_sym_(A, diag, offdiag);

  // QR iteration on tridiagonal matrix
  mat__tridiag_qr_iter_(diag, offdiag, n, NULL, 0);

  // Copy eigenvalues to output
  memcpy(out->data, diag, n * sizeof(mat_elem_t));

  MAT_FREE(offdiag);
  MAT_FREE(diag);
}

MATDEF void mat_eigvals_sym(Vec *out, const Mat *A) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(A);
  MAT_ASSERT(A->rows == A->cols);
  MAT_ASSERT(out->rows == A->rows && out->cols == 1);

  mat__eigvals_sym_scalar_(out, A);
}

/* Symmetric Eigendecomposition */

// Scalar implementation of symmetric eigendecomposition
MAT_INTERNAL_STATIC void mat__eigen_sym_scalar_(Mat *V, Vec *eigenvalues, const Mat *A) {
  size_t n = A->rows;

  // Handle 1x1 case
  if (n == 1) {
    eigenvalues->data[0] = MAT_AT(A, 0, 0);
    V->data[0] = 1;
    return;
  }

  // Handle 2x2 case directly
  if (n == 2) {
    mat_elem_t a = MAT_AT(A, 0, 0);
    mat_elem_t b = MAT_AT(A, 0, 1);
    mat_elem_t d = MAT_AT(A, 1, 1);

    mat_elem_t trace = a + d;
    mat_elem_t det = a * d - b * b;
    mat_elem_t disc = trace * trace - 4 * det;
    mat_elem_t sqrt_disc = (disc > 0) ? MAT_SQRT(disc) : 0;

    // lambda_small < lambda_large (ascending order)
    mat_elem_t lambda_small = (trace - sqrt_disc) / 2;
    mat_elem_t lambda_large = (trace + sqrt_disc) / 2;

    // Compute eigenvectors (column 0 for small eigenvalue, column 1 for large)
    if (MAT_FABS(b) > MAT_DEFAULT_EPSILON) {
      // Eigenvector for lambda_small
      mat_elem_t v1_x = lambda_small - d;
      mat_elem_t v1_y = b;
      mat_elem_t norm1 = MAT_SQRT(v1_x * v1_x + v1_y * v1_y);
      MAT_SET(V, 0, 0, v1_x / norm1);
      MAT_SET(V, 1, 0, v1_y / norm1);

      // Eigenvector for lambda_large
      mat_elem_t v2_x = lambda_large - d;
      mat_elem_t v2_y = b;
      mat_elem_t norm2 = MAT_SQRT(v2_x * v2_x + v2_y * v2_y);
      MAT_SET(V, 0, 1, v2_x / norm2);
      MAT_SET(V, 1, 1, v2_y / norm2);
    } else {
      // Diagonal case: eigenvalues are a and d
      if (a <= d) {
        MAT_SET(V, 0, 0, 1); MAT_SET(V, 1, 0, 0);
        MAT_SET(V, 0, 1, 0); MAT_SET(V, 1, 1, 1);
      } else {
        MAT_SET(V, 0, 0, 0); MAT_SET(V, 1, 0, 1);
        MAT_SET(V, 0, 1, 1); MAT_SET(V, 1, 1, 0);
      }
    }

    eigenvalues->data[0] = lambda_small;
    eigenvalues->data[1] = lambda_large;
    return;
  }

  // Allocate working storage
  mat_elem_t *W = (mat_elem_t *)MAT_MALLOC(n * n * sizeof(mat_elem_t));
  mat_elem_t *diag = (mat_elem_t *)MAT_MALLOC(n * sizeof(mat_elem_t));
  mat_elem_t *offdiag = (mat_elem_t *)MAT_MALLOC((n - 1) * sizeof(mat_elem_t));
  mat_elem_t *v = (mat_elem_t *)MAT_MALLOC(n * sizeof(mat_elem_t));
  mat_elem_t *p = (mat_elem_t *)MAT_MALLOC(n * sizeof(mat_elem_t));

  memcpy(W, A->data, n * n * sizeof(mat_elem_t));

  // Initialize V to identity (will accumulate Q)
  memset(V->data, 0, n * n * sizeof(mat_elem_t));
  for (size_t i = 0; i < n; i++) {
    MAT_SET(V, i, i, 1);
  }

  // Reduce to tridiagonal form, accumulating Q in V
  // TODO: Implement blocked tridiagonalization with Q accumulation (WY representation)
  // to match Eigen's performance for n >= 64. Currently ~1.7x slower than Eigen at n=256.
  mat__tridiag_sym_unblocked_(W, n, diag, offdiag, v, p, V->data, n);

  // QR iteration on tridiagonal, accumulating rotations in V
  mat__tridiag_qr_iter_(diag, offdiag, n, V->data, n);

  MAT_FREE(p);
  MAT_FREE(v);

  // Copy eigenvalues
  memcpy(eigenvalues->data, diag, n * sizeof(mat_elem_t));

  // Sort eigenvalues ascending and permute eigenvectors
  for (size_t i = 0; i < n - 1; i++) {
    size_t min_idx = i;
    for (size_t j = i + 1; j < n; j++) {
      if (eigenvalues->data[j] < eigenvalues->data[min_idx]) {
        min_idx = j;
      }
    }
    if (min_idx != i) {
      // Swap eigenvalues
      mat_elem_t tmp = eigenvalues->data[i];
      eigenvalues->data[i] = eigenvalues->data[min_idx];
      eigenvalues->data[min_idx] = tmp;

      // Swap eigenvector columns
      mat_elem_t *col_i = &V->data[i * n];
      mat_elem_t *col_min = &V->data[min_idx * n];
      for (size_t k = 0; k < n; k++) {
        mat_elem_t t = col_i[k];
        col_i[k] = col_min[k];
        col_min[k] = t;
      }
    }
  }

  MAT_FREE(offdiag);
  MAT_FREE(diag);
  MAT_FREE(W);
}

// Symmetric eigendecomposition: A = V * diag(eigenvalues) * V^T
// V contains eigenvectors as columns (orthogonal matrix)
// Eigenvalues are sorted in ascending order
MATDEF void mat_eigen_sym(Mat *V, Vec *eigenvalues, const Mat *A) {
  MAT_ASSERT_MAT(V);
  MAT_ASSERT_MAT(eigenvalues);
  MAT_ASSERT_MAT(A);
  MAT_ASSERT(A->rows == A->cols);
  MAT_ASSERT(V->rows == A->rows && V->cols == A->cols);
  MAT_ASSERT(eigenvalues->rows == A->rows && eigenvalues->cols == 1);

  mat__eigen_sym_scalar_(V, eigenvalues, A);
}

// Scalar implementation of eigenvalue computation
MAT_INTERNAL_STATIC void mat__eigvals_scalar_(Vec *out, const Mat *A) {
  size_t n = A->rows;

  // Handle small cases
  if (n == 1) {
    out->data[0] = MAT_AT(A, 0, 0);
    return;
  }

  if (n == 2) {
    mat_elem_t a = MAT_AT(A, 0, 0);
    mat_elem_t b = MAT_AT(A, 0, 1);
    mat_elem_t c = MAT_AT(A, 1, 0);
    mat_elem_t d = MAT_AT(A, 1, 1);

    mat_elem_t trace = a + d;
    mat_elem_t det = a * d - b * c;
    mat_elem_t disc = trace * trace - 4 * det;

    if (disc >= 0) {
      mat_elem_t sqrt_disc = MAT_SQRT(disc);
      out->data[0] = (trace + sqrt_disc) / 2;
      out->data[1] = (trace - sqrt_disc) / 2;
    } else {
      // Complex eigenvalues: store real parts
      out->data[0] = trace / 2;
      out->data[1] = trace / 2;
    }
    return;
  }

  // Copy A to working matrix H
  Mat *H = mat_rdeep_copy(A);

  // Reduce to upper Hessenberg form
  mat__hessenberg_(H, NULL, 0);

  // QR iteration to reduce to quasi-triangular form
  mat__multishift_qr_iter_(H, NULL, 0);

  // Extract eigenvalues from quasi-triangular form
  mat__extract_eigvals_(out, H, 0, n - 1);

  MAT_FREE_MAT(H);
}


MATDEF void mat_eigvals(Vec *out, const Mat *A) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(A);
  MAT_ASSERT(A->rows == A->cols);
  MAT_ASSERT(out->rows == A->rows && out->cols == 1);

  mat__eigvals_scalar_(out, A);
}

// Compute eigendecomposition of general (non-symmetric) matrix A.
// V: n x n output matrix of eigenvectors (columns)
// eigenvalues: n x 1 output vector of eigenvalues (real parts for complex)
// A: n x n input matrix (not modified)
//
// For real eigenvalues: V[:, k] is the corresponding eigenvector.
// For complex conjugate pairs at positions k, k+1:
//   - eigenvalues[k] = eigenvalues[k+1] = real part
//   - V[:, k] = real part of eigenvector
//   - V[:, k+1] = imaginary part of eigenvector
//   - Actual eigenvectors are: V[:, k] ± i * V[:, k+1]
MAT_INTERNAL_STATIC void mat__eigen_scalar_(Mat *V, Vec *eigenvalues, const Mat *A) {
  size_t n = A->rows;

  // Handle small cases
  if (n == 1) {
    eigenvalues->data[0] = MAT_AT(A, 0, 0);
    MAT_SET(V, 0, 0, 1.0);
    return;
  }

  if (n == 2) {
    mat_elem_t a = MAT_AT(A, 0, 0);
    mat_elem_t b = MAT_AT(A, 0, 1);
    mat_elem_t c = MAT_AT(A, 1, 0);
    mat_elem_t d = MAT_AT(A, 1, 1);

    mat_elem_t trace = a + d;
    mat_elem_t det = a * d - b * c;
    mat_elem_t disc = trace * trace - 4 * det;

    if (disc >= 0) {
      // Real eigenvalues
      mat_elem_t sqrt_disc = MAT_SQRT(disc);
      mat_elem_t lambda1 = (trace + sqrt_disc) / 2;
      mat_elem_t lambda2 = (trace - sqrt_disc) / 2;
      eigenvalues->data[0] = lambda1;
      eigenvalues->data[1] = lambda2;

      // Eigenvectors: (A - λI) * v = 0
      // For λ1: if c != 0, v = [λ1 - d, c], else v = [b, λ1 - a] or [1, 0]
      if (MAT_FABS(c) > MAT_DEFAULT_EPSILON) {
        mat_elem_t norm1 = MAT_SQRT((lambda1 - d) * (lambda1 - d) + c * c);
        MAT_SET(V, 0, 0, (lambda1 - d) / norm1);
        MAT_SET(V, 1, 0, c / norm1);
        mat_elem_t norm2 = MAT_SQRT((lambda2 - d) * (lambda2 - d) + c * c);
        MAT_SET(V, 0, 1, (lambda2 - d) / norm2);
        MAT_SET(V, 1, 1, c / norm2);
      } else if (MAT_FABS(b) > MAT_DEFAULT_EPSILON) {
        mat_elem_t norm1 = MAT_SQRT(b * b + (lambda1 - a) * (lambda1 - a));
        MAT_SET(V, 0, 0, b / norm1);
        MAT_SET(V, 1, 0, (lambda1 - a) / norm1);
        mat_elem_t norm2 = MAT_SQRT(b * b + (lambda2 - a) * (lambda2 - a));
        MAT_SET(V, 0, 1, b / norm2);
        MAT_SET(V, 1, 1, (lambda2 - a) / norm2);
      } else {
        // Diagonal matrix
        MAT_SET(V, 0, 0, 1.0);
        MAT_SET(V, 1, 0, 0.0);
        MAT_SET(V, 0, 1, 0.0);
        MAT_SET(V, 1, 1, 1.0);
      }
    } else {
      // Complex conjugate eigenvalues
      eigenvalues->data[0] = trace / 2;
      eigenvalues->data[1] = trace / 2;
      mat_elem_t im = MAT_SQRT(-disc) / 2;

      // Eigenvector: real part in col 0, imag part in col 1
      if (MAT_FABS(c) > MAT_DEFAULT_EPSILON) {
        mat_elem_t re_part = (trace / 2 - d);
        mat_elem_t norm = MAT_SQRT(re_part * re_part + im * im + c * c);
        MAT_SET(V, 0, 0, re_part / norm);
        MAT_SET(V, 0, 1, im / norm);
        MAT_SET(V, 1, 0, c / norm);
        MAT_SET(V, 1, 1, 0.0);
      } else {
        MAT_SET(V, 0, 0, 1.0);
        MAT_SET(V, 0, 1, 0.0);
        MAT_SET(V, 1, 0, 0.0);
        MAT_SET(V, 1, 1, 1.0);
      }
    }
    return;
  }

  // Copy A to working matrix H
  Mat *H = mat_rdeep_copy(A);

  // Initialize Q to identity for accumulating Hessenberg transformations
  Mat *Q = mat_mat(n, n);
  mat_eye(Q);

  // Reduce to upper Hessenberg form, accumulating Q
  mat__hessenberg_(H, Q->data, n);

  // QR iteration, accumulating transformations in Q
  mat__multishift_qr_iter_(H, Q->data, n);

  // Extract eigenvalues from quasi-triangular Schur form H
  mat__extract_eigvals_(eigenvalues, H, 0, n - 1);

  // Compute eigenvectors of Schur form H
  Mat *Y = mat_mat(n, n);
  mat__trevc_(Y, H);

  // Back-transform: V = Q * Y
  mat_mul(V, Q, Y);

  MAT_FREE_MAT(H);
  MAT_FREE_MAT(Q);
  MAT_FREE_MAT(Y);
}

MATDEF void mat_eigen(Mat *V, Vec *eigenvalues, const Mat *A) {
  MAT_ASSERT_MAT(V);
  MAT_ASSERT_MAT(eigenvalues);
  MAT_ASSERT_MAT(A);
  MAT_ASSERT(A->rows == A->cols);
  MAT_ASSERT(V->rows == A->rows && V->cols == A->cols);
  MAT_ASSERT(eigenvalues->rows == A->rows && eigenvalues->cols == 1);

  mat__eigen_scalar_(V, eigenvalues, A);
}

#endif // MAT_IMPLEMENTATION
