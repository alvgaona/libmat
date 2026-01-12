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

// OpenMP detection (auto-enabled when compiled with -fopenmp)
#if defined(_OPENMP)
  #include <omp.h>
  #define MAT_HAS_OPENMP
  #ifndef MAT_OMP_THRESHOLD
    #define MAT_OMP_THRESHOLD (1024 * 1024)  // Skip parallelization for small matrices
  #endif
#endif

#ifndef MATDEF
#define MATDEF
#endif

// Mark experimental functions (warns on use)
#if defined(__GNUC__)
#define MAT_EXPERIMENTAL __attribute__((warning("experimental: does not guarantee correct results")))
#elif defined(_MSC_VER)
#define MAT_EXPERIMENTAL __declspec(deprecated("experimental: does not guarantee correct results"))
#else
#define MAT_EXPERIMENTAL
#endif

// Mark unimplemented functions (errors on use)
#if defined(__GNUC__)
#define MAT_NOT_IMPLEMENTED __attribute__((error("not implemented")))
#elif defined(_MSC_VER)
#define MAT_NOT_IMPLEMENTED __declspec(deprecated("not implemented - will not link"))
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
#define MAT_FREE_MAT(m) do { MAT_FREE((m)->data); MAT_FREE(m); } while(0)
#endif

#ifndef MAT_FREE_PERM
#define MAT_FREE_PERM(p) do { MAT_FREE((p)->data); MAT_FREE(p); } while(0)
#endif

// Scratch arena for temporary allocations in hot paths
// Define MAT_NO_SCRATCH to disable (uses malloc/free instead)
// Define MAT_SCRATCH_SIZE to override the default size
#ifndef MAT_SCRATCH_SIZE
#define MAT_SCRATCH_SIZE (4 * 1024 * 1024)  // 4MB default
#endif

#ifndef MAT_NO_SCRATCH
typedef struct {
  char *buf;
  size_t offset;
  size_t size;
} MatArena;
#endif

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
  #define MAT_FABS fabs
  #define MAT_SQRT sqrt
#else
  typedef float mat_elem_t;
  #ifndef MAT_DEFAULT_EPSILON
    #define MAT_DEFAULT_EPSILON 1e-6f
  #endif
  #define MAT_FABS fabsf
  #define MAT_SQRT sqrtf
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
    #define MAT_NEON_FMS      vfmsq_f64
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
    #define MAT_NEON_FMS      vfmsq_f32
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

  // Accumulator macros for overflow-safe reductions (always use double)
  #define MAT_ACC_TYPE       float64x2_t
  #define MAT_ACC_ZERO       vdupq_n_f64(0)
  #define MAT_ACC_ADD        vaddq_f64
  #define MAT_ACC_ADDV       vaddvq_f64
  #define MAT_ACC_FMA        vfmaq_f64
  #ifdef MAT_DOUBLE_PRECISION
    #define MAT_ACC_WIDTH    2
    #define MAT_ACC_LOAD_SQ(acc, ptr) do { \
        float64x2_t _v = vld1q_f64(ptr); \
        acc = vfmaq_f64(acc, _v, _v); \
      } while(0)
  #else
    #define MAT_ACC_WIDTH    2
    #define MAT_ACC_LOAD_SQ(acc, ptr) do { \
        float32x2_t _v = vld1_f32(ptr); \
        float64x2_t _d = vcvt_f64_f32(_v); \
        acc = vfmaq_f64(acc, _d, _d); \
      } while(0)
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

typedef struct {
  size_t *data;
  size_t size;
} Perm;

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

// Set matrix to identity (must be square).
MATDEF void mat_eye(Mat *out);

// Create identity matrix of given dimension.
MATDEF Mat *mat_reye(size_t dim);

// Allocate zero-initialized column vector.
MATDEF Vec *mat_vec(size_t dim);

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

// Shallow copy (copies struct, shares data pointer). Use mat_rdeep_copy for full copy.
MATDEF Mat *mat_copy(const Mat *m);

// Deep copy src into pre-allocated out.
MATDEF void mat_deep_copy(Mat *out, const Mat *src);

// Allocate and return deep copy.
MATDEF Mat *mat_rdeep_copy(const Mat *m);

/* Accessors & Info */

// Get element at (row, col). Bounds checked via MAT_ASSERT.
MATDEF mat_elem_t mat_at(const Mat *mat, size_t row, size_t col);

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
MATDEF void mat_clip(Mat *out, const Mat *a, mat_elem_t min_val, mat_elem_t max_val);

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

// out = v1 * v2^T (outer product). out(m,n) where v1 is m-dim, v2 is n-dim. SIMD-optimized.
MATDEF void mat_outer(Mat *out, const Vec *v1, const Vec *v2);

// Return x^T * A * y (bilinear form). x is m-dim, A is m x n, y is n-dim.
MATDEF mat_elem_t mat_bilinear(const Vec *x, const Mat *A, const Vec *y);

// Return x^T * A * x (quadratic form). x is n-dim, A is n x n.
MATDEF mat_elem_t mat_quadform(const Vec *x, const Mat *A);

// Fused Operations (BLAS-like)

// y = alpha * x + y (AXPY). SIMD-optimized.
MATDEF void mat_axpy(Vec *y, mat_elem_t alpha, const Vec *x);

// y = alpha * A * x + beta * y (GEMV). SIMD-optimized.
MATDEF void mat_gemv(Vec *y, mat_elem_t alpha, const Mat *A, const Vec *x, mat_elem_t beta);

// A = A + alpha * x * y^T (GER/rank-1 update).
MATDEF void mat_ger(Mat *A, mat_elem_t alpha, const Vec *x, const Vec *y);

// C = alpha * A * B + beta * C (GEMM). SIMD-optimized.
MATDEF void mat_gemm(Mat *C, mat_elem_t alpha, const Mat *A, const Mat *B, mat_elem_t beta);

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

// Extract submatrix m[row_start:row_end, col_start:col_end]. Allocates new matrix.
// Indices are inclusive start, exclusive end.
MATDEF Mat *mat_slice(const Mat *m, size_t row_start, size_t row_end, size_t col_start, size_t col_end);

// Copy src into m starting at (row_start, col_start).
MATDEF void mat_slice_set(Mat *m, size_t row_start, size_t col_start, const Mat *src);

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
// Overflows if any |a_ij|^2 exceeds type max (~1e19 for float, ~1e154 for double).
MATDEF mat_elem_t mat_norm_fro_fast(const Mat *a);

// Matrix Properties

// Return trace (sum of diagonal elements). Matrix must be square.
MATDEF mat_elem_t mat_trace(const Mat *a);

// Return determinant. Matrix must be square. Uses LU decomposition.
MATDEF mat_elem_t mat_det(const Mat *a);

// Return numerical rank via SVD.
MAT_NOT_IMPLEMENTED MATDEF mat_elem_t mat_rank(const Mat *a);

// Return condition number.
MAT_NOT_IMPLEMENTED MATDEF mat_elem_t mat_cond(const Mat *a);

// Return count of non-zero elements.
MATDEF mat_elem_t mat_nnz(const Mat *a);

// Decomposition

// QR decomposition via Householder reflections.
// A = Q * R where Q is orthogonal (m x m), R is upper triangular (m x n).
// Q and R must be pre-allocated with correct dimensions.
MAT_EXPERIMENTAL MATDEF void mat_qr(const Mat *A, Mat *Q, Mat *R);

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
MAT_NOT_IMPLEMENTED MATDEF void mat_chol(const Mat *A, Mat *L);

// Singular value decomposition (A = U * S * Vt).
MAT_NOT_IMPLEMENTED MATDEF void mat_svd(const Mat *A, Mat *U, Vec *S, Mat *Vt);

// Matrix inverse using LU decomposition.
MATDEF void mat_inv(Mat *out, const Mat *A);

// Moore-Penrose pseudoinverse via SVD.
MAT_NOT_IMPLEMENTED MATDEF void mat_pinv(Mat *out, const Mat *A);

// Eigendecomposition.
MAT_NOT_IMPLEMENTED MATDEF void mat_eig(const Mat *A, Vec *eigenvalues, Mat *eigenvectors);

// Eigenvalues only (faster, no eigenvectors).
MAT_NOT_IMPLEMENTED MATDEF void mat_eigvals(Vec *out, const Mat *A);

// Solve Ax = b.
MAT_NOT_IMPLEMENTED MATDEF void mat_solve(Vec *x, const Mat *A, const Vec *b);

// Least squares solution.
MAT_NOT_IMPLEMENTED MATDEF void mat_lstsq(Vec *x, const Mat *A, const Vec *b);

// Kronecker product.
MAT_NOT_IMPLEMENTED MATDEF void mat_kron(Mat *out, const Mat *A, const Mat *B);

// 2D convolution.
MAT_NOT_IMPLEMENTED MATDEF void mat_conv2d(Mat *out, const Mat *A, const Mat *kernel);

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
  bytes = (bytes + 15) & ~15;  // align to 16 bytes for NEON
  if (mat_scratch_.offset + bytes > mat_scratch_.size) {
    // Fall back to heap if arena is full
    return MAT_MALLOC(bytes);
  }
  void *ptr = mat_scratch_.buf + mat_scratch_.offset;
  mat_scratch_.offset += bytes;
  return ptr;
}

static inline void mat_scratch_reset_(void) {
  mat_scratch_.offset = 0;
}
#else
// No scratch arena - use malloc/free directly
static inline void *mat_scratch_alloc_(size_t bytes) {
  return MAT_MALLOC(bytes);
}
static inline void mat_scratch_free_(void *ptr) {
  MAT_FREE(ptr);
}
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

  for (size_t i = 0; i < out->rows; i++) {
    for (size_t j = 0; j < out->cols; j++) {
      out->data[i * out->cols + j] = values[i * out->cols + j];
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

  for (size_t i = 0; i < rows * cols; i++)
    result->data[i] = 1;

  return result;
}

MATDEF void mat_eye(Mat *out) {
  MAT_ASSERT_SQUARE(out);
  size_t dim = out->rows;

  memset(out->data, 0, dim * dim * sizeof(mat_elem_t));
  for (size_t i = 0; i < dim; i++) {
    out->data[i * dim + i] = 1;
  }
}

MATDEF Mat *mat_reye(size_t dim) {
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
    m->data[i * n + p->data[i]] = 1;
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

/* Element-wise Unary */

MATDEF void mat_abs(Mat *out, const Mat *a) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);

  size_t len = a->rows * a->cols;
  for (size_t i = 0; i < len; i++) out->data[i] = fabs(a->data[i]);
}

MATDEF void mat_sqrt(Mat *out, const Mat *a) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);

  size_t len = a->rows * a->cols;
#ifdef MAT_DOUBLE_PRECISION
  for (size_t i = 0; i < len; i++) out->data[i] = sqrt(a->data[i]);
#else
  for (size_t i = 0; i < len; i++) out->data[i] = sqrtf(a->data[i]);
#endif
}

MATDEF void mat_exp(Mat *out, const Mat *a) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);

  size_t len = a->rows * a->cols;
#ifdef MAT_DOUBLE_PRECISION
  for (size_t i = 0; i < len; i++) out->data[i] = exp(a->data[i]);
#else
  for (size_t i = 0; i < len; i++) out->data[i] = expf(a->data[i]);
#endif
}

MATDEF void mat_log(Mat *out, const Mat *a) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);

  size_t len = a->rows * a->cols;
#ifdef MAT_DOUBLE_PRECISION
  for (size_t i = 0; i < len; i++) out->data[i] = log(a->data[i]);
#else
  for (size_t i = 0; i < len; i++) out->data[i] = logf(a->data[i]);
#endif
}

MATDEF void mat_log10(Mat *out, const Mat *a) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);

  size_t len = a->rows * a->cols;
#ifdef MAT_DOUBLE_PRECISION
  for (size_t i = 0; i < len; i++) out->data[i] = log10(a->data[i]);
#else
  for (size_t i = 0; i < len; i++) out->data[i] = log10f(a->data[i]);
#endif
}

MATDEF void mat_sin(Mat *out, const Mat *a) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);

  size_t len = a->rows * a->cols;
#ifdef MAT_DOUBLE_PRECISION
  for (size_t i = 0; i < len; i++) out->data[i] = sin(a->data[i]);
#else
  for (size_t i = 0; i < len; i++) out->data[i] = sinf(a->data[i]);
#endif
}

MATDEF void mat_cos(Mat *out, const Mat *a) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);

  size_t len = a->rows * a->cols;
#ifdef MAT_DOUBLE_PRECISION
  for (size_t i = 0; i < len; i++) out->data[i] = cos(a->data[i]);
#else
  for (size_t i = 0; i < len; i++) out->data[i] = cosf(a->data[i]);
#endif
}

MATDEF void mat_pow(Mat *out, const Mat *a, mat_elem_t exponent) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);

  size_t len = a->rows * a->cols;
#ifdef MAT_DOUBLE_PRECISION
  for (size_t i = 0; i < len; i++) out->data[i] = pow(a->data[i], exponent);
#else
  for (size_t i = 0; i < len; i++) out->data[i] = powf(a->data[i], exponent);
#endif
}

MATDEF void mat_clip(Mat *out, const Mat *a, mat_elem_t min_val, mat_elem_t max_val) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);

  size_t len = a->rows * a->cols;
  for (size_t i = 0; i < len; i++) {
    mat_elem_t v = a->data[i];
    if (v < min_val) v = min_val;
    else if (v > max_val) v = max_val;
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
  for (size_t i = 0; i < len; i++) out->data[i] = atan2(y->data[i], x->data[i]);
#else
  for (size_t i = 0; i < len; i++) out->data[i] = atan2f(y->data[i], x->data[i]);
#endif
}

/* Scalar Operations */

MATDEF void mat_scale(Mat *out, mat_elem_t k) {
  MAT_ASSERT_MAT(out);

  for (size_t i = 0; i < out->rows * out->cols; i++) {
    out->data[i] *= k;
  }
}

MATDEF Mat *mat_rscale(const Mat *m, mat_elem_t k) {
  Mat *result = mat_rdeep_copy(m);

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

  Mat *result = mat_rdeep_copy(first);

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
MAT_INTERNAL_STATIC void mat_outer_neon_impl(Mat *out, const Vec *v1, const Vec *v2) {
  size_t m = v1->rows * v1->cols;
  size_t n = v2->rows * v2->cols;

  const mat_elem_t *a = v1->data;
  const mat_elem_t *b = v2->data;
  mat_elem_t *c = out->data;

  for (size_t i = 0; i < m; i++) {
    MAT_NEON_TYPE va = MAT_NEON_DUP(a[i]);
    mat_elem_t *row = &c[i * n];
    size_t j = 0;

    for (; j + MAT_NEON_WIDTH * 4 <= n; j += MAT_NEON_WIDTH * 4) {
      MAT_NEON_TYPE vb0 = MAT_NEON_LOAD(&b[j]);
      MAT_NEON_TYPE vb1 = MAT_NEON_LOAD(&b[j + MAT_NEON_WIDTH]);
      MAT_NEON_TYPE vb2 = MAT_NEON_LOAD(&b[j + MAT_NEON_WIDTH * 2]);
      MAT_NEON_TYPE vb3 = MAT_NEON_LOAD(&b[j + MAT_NEON_WIDTH * 3]);
#ifdef MAT_DOUBLE_PRECISION
      MAT_NEON_STORE(&row[j], vmulq_f64(va, vb0));
      MAT_NEON_STORE(&row[j + MAT_NEON_WIDTH], vmulq_f64(va, vb1));
      MAT_NEON_STORE(&row[j + MAT_NEON_WIDTH * 2], vmulq_f64(va, vb2));
      MAT_NEON_STORE(&row[j + MAT_NEON_WIDTH * 3], vmulq_f64(va, vb3));
#else
      MAT_NEON_STORE(&row[j], vmulq_f32(va, vb0));
      MAT_NEON_STORE(&row[j + MAT_NEON_WIDTH], vmulq_f32(va, vb1));
      MAT_NEON_STORE(&row[j + MAT_NEON_WIDTH * 2], vmulq_f32(va, vb2));
      MAT_NEON_STORE(&row[j + MAT_NEON_WIDTH * 3], vmulq_f32(va, vb3));
#endif
    }

    for (; j + MAT_NEON_WIDTH <= n; j += MAT_NEON_WIDTH) {
      MAT_NEON_TYPE vb = MAT_NEON_LOAD(&b[j]);
#ifdef MAT_DOUBLE_PRECISION
      MAT_NEON_STORE(&row[j], vmulq_f64(va, vb));
#else
      MAT_NEON_STORE(&row[j], vmulq_f32(va, vb));
#endif
    }

    for (; j < n; j++) {
      row[j] = a[i] * b[j];
    }
  }
}
#endif

MAT_INTERNAL_STATIC void mat_outer_scalar_impl(Mat *out, const Vec *v1, const Vec *v2) {
  size_t m = v1->rows * v1->cols;
  size_t n = v2->rows * v2->cols;

  const mat_elem_t *a = v1->data;
  const mat_elem_t *b = v2->data;

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      out->data[i * n + j] = a[i] * b[j];
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
  mat_outer_neon_impl(out, v1, v2);
#else
  mat_outer_scalar_impl(out, v1, v2);
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
MAT_INTERNAL_STATIC void mat_axpy_neon_impl(Vec *y, mat_elem_t alpha, const Vec *x) {
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

MAT_INTERNAL_STATIC void mat_axpy_scalar_impl(Vec *y, mat_elem_t alpha, const Vec *x) {
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
  mat_axpy_neon_impl(y, alpha, x);
#else
  mat_axpy_scalar_impl(y, alpha, x);
#endif
}

// y = alpha * A * x + beta * y (BLAS Level-2: gemv)
#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void mat_gemv_neon_impl(Vec *y, mat_elem_t alpha, const Mat *A, const Vec *x, mat_elem_t beta) {
  size_t m = A->rows;
  size_t n = A->cols;
  mat_elem_t *py = y->data;
  const mat_elem_t *pa = A->data;
  const mat_elem_t *px = x->data;

  for (size_t i = 0; i < m; i++) {
    MAT_NEON_TYPE vsum0 = MAT_NEON_DUP(0);
    MAT_NEON_TYPE vsum1 = MAT_NEON_DUP(0);
    MAT_NEON_TYPE vsum2 = MAT_NEON_DUP(0);
    MAT_NEON_TYPE vsum3 = MAT_NEON_DUP(0);
    const mat_elem_t *row = &pa[i * n];

    size_t j = 0;
    for (; j + MAT_NEON_WIDTH * 4 <= n; j += MAT_NEON_WIDTH * 4) {
      MAT_NEON_TYPE va0 = MAT_NEON_LOAD(&row[j]);
      MAT_NEON_TYPE va1 = MAT_NEON_LOAD(&row[j + MAT_NEON_WIDTH]);
      MAT_NEON_TYPE va2 = MAT_NEON_LOAD(&row[j + MAT_NEON_WIDTH * 2]);
      MAT_NEON_TYPE va3 = MAT_NEON_LOAD(&row[j + MAT_NEON_WIDTH * 3]);
      MAT_NEON_TYPE vx0 = MAT_NEON_LOAD(&px[j]);
      MAT_NEON_TYPE vx1 = MAT_NEON_LOAD(&px[j + MAT_NEON_WIDTH]);
      MAT_NEON_TYPE vx2 = MAT_NEON_LOAD(&px[j + MAT_NEON_WIDTH * 2]);
      MAT_NEON_TYPE vx3 = MAT_NEON_LOAD(&px[j + MAT_NEON_WIDTH * 3]);
      vsum0 = MAT_NEON_FMA(vsum0, va0, vx0);
      vsum1 = MAT_NEON_FMA(vsum1, va1, vx1);
      vsum2 = MAT_NEON_FMA(vsum2, va2, vx2);
      vsum3 = MAT_NEON_FMA(vsum3, va3, vx3);
    }

    for (; j + MAT_NEON_WIDTH <= n; j += MAT_NEON_WIDTH) {
      MAT_NEON_TYPE va = MAT_NEON_LOAD(&row[j]);
      MAT_NEON_TYPE vx = MAT_NEON_LOAD(&px[j]);
      vsum0 = MAT_NEON_FMA(vsum0, va, vx);
    }

    vsum0 = MAT_NEON_ADD(vsum0, vsum1);
    vsum2 = MAT_NEON_ADD(vsum2, vsum3);
    vsum0 = MAT_NEON_ADD(vsum0, vsum2);
    mat_elem_t sum = MAT_NEON_ADDV(vsum0);

    for (; j < n; j++) {
      sum += row[j] * px[j];
    }

    py[i] = alpha * sum + beta * py[i];
  }
}
#endif

MAT_INTERNAL_STATIC void mat_gemv_scalar_impl(Vec *y, mat_elem_t alpha, const Mat *A, const Vec *x, mat_elem_t beta) {
  size_t m = A->rows;
  size_t n = A->cols;

  for (size_t i = 0; i < m; i++) {
    mat_elem_t sum = 0;
    for (size_t j = 0; j < n; j++) {
      sum += A->data[i * n + j] * x->data[j];
    }
    y->data[i] = alpha * sum + beta * y->data[i];
  }
}

MATDEF void mat_gemv(Vec *y, mat_elem_t alpha, const Mat *A, const Vec *x, mat_elem_t beta) {
  MAT_ASSERT_MAT(y);
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(x);
  MAT_ASSERT(A->rows == y->rows);
  MAT_ASSERT(A->cols == x->rows);

#ifdef MAT_HAS_ARM_NEON
  mat_gemv_neon_impl(y, alpha, A, x, beta);
#else
  mat_gemv_scalar_impl(y, alpha, A, x, beta);
#endif
}

// A += alpha * x * y^T (BLAS Level-2: ger - rank-1 update)
#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void mat_ger_neon_impl(Mat *A, mat_elem_t alpha, const Vec *x, const Vec *y) {
  size_t m = A->rows;
  size_t n = A->cols;
  mat_elem_t *pa = A->data;
  const mat_elem_t *px = x->data;
  const mat_elem_t *py = y->data;

  for (size_t i = 0; i < m; i++) {
    mat_elem_t *row = &pa[i * n];
    MAT_NEON_TYPE vxi = MAT_NEON_DUP(alpha * px[i]);

    size_t j = 0;
    for (; j + MAT_NEON_WIDTH * 4 <= n; j += MAT_NEON_WIDTH * 4) {
      MAT_NEON_TYPE va0 = MAT_NEON_LOAD(&row[j]);
      MAT_NEON_TYPE va1 = MAT_NEON_LOAD(&row[j + MAT_NEON_WIDTH]);
      MAT_NEON_TYPE va2 = MAT_NEON_LOAD(&row[j + MAT_NEON_WIDTH * 2]);
      MAT_NEON_TYPE va3 = MAT_NEON_LOAD(&row[j + MAT_NEON_WIDTH * 3]);
      MAT_NEON_TYPE vy0 = MAT_NEON_LOAD(&py[j]);
      MAT_NEON_TYPE vy1 = MAT_NEON_LOAD(&py[j + MAT_NEON_WIDTH]);
      MAT_NEON_TYPE vy2 = MAT_NEON_LOAD(&py[j + MAT_NEON_WIDTH * 2]);
      MAT_NEON_TYPE vy3 = MAT_NEON_LOAD(&py[j + MAT_NEON_WIDTH * 3]);
      va0 = MAT_NEON_FMA(va0, vy0, vxi);
      va1 = MAT_NEON_FMA(va1, vy1, vxi);
      va2 = MAT_NEON_FMA(va2, vy2, vxi);
      va3 = MAT_NEON_FMA(va3, vy3, vxi);
      MAT_NEON_STORE(&row[j], va0);
      MAT_NEON_STORE(&row[j + MAT_NEON_WIDTH], va1);
      MAT_NEON_STORE(&row[j + MAT_NEON_WIDTH * 2], va2);
      MAT_NEON_STORE(&row[j + MAT_NEON_WIDTH * 3], va3);
    }

    for (; j + MAT_NEON_WIDTH <= n; j += MAT_NEON_WIDTH) {
      MAT_NEON_TYPE va = MAT_NEON_LOAD(&row[j]);
      MAT_NEON_TYPE vy = MAT_NEON_LOAD(&py[j]);
      va = MAT_NEON_FMA(va, vy, vxi);
      MAT_NEON_STORE(&row[j], va);
    }

    mat_elem_t xi = alpha * px[i];
    for (; j < n; j++) {
      row[j] += xi * py[j];
    }
  }
}
#endif

MAT_INTERNAL_STATIC void mat_ger_scalar_impl(Mat *A, mat_elem_t alpha, const Vec *x, const Vec *y) {
  size_t m = A->rows;
  size_t n = A->cols;

  for (size_t i = 0; i < m; i++) {
    mat_elem_t xi = alpha * x->data[i];
    for (size_t j = 0; j < n; j++) {
      A->data[i * n + j] += xi * y->data[j];
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
  mat_ger_neon_impl(A, alpha, x, y);
#else
  mat_ger_scalar_impl(A, alpha, x, y);
#endif
}

// Unit lower triangular GEMM for QR decomposition
// V is unit lower triangular stored column-major: V[col * ldv + row] for row > col
// Implicit 1 on diagonal, implicit 0 above diagonal

#ifdef MAT_HAS_ARM_NEON
// W = V^T * R where V is unit lower triangular (panel_rows x kb), R is (panel_rows x N)
// Result W is (kb x N), stored row-major with stride ldw
// V stored column-major: V[col * ldv + row] for row > col
MAT_INTERNAL_STATIC void mat_gemm_unit_lower_t_neon(
    mat_elem_t *W, size_t ldw,
    const mat_elem_t *V, size_t ldv,
    const mat_elem_t *R, size_t ldr,
    size_t kb, size_t panel_rows, size_t N) {

  for (size_t ii = 0; ii < kb; ii++) {
    mat_elem_t *Wi = &W[ii * ldw];
    const mat_elem_t *Rii = &R[ii * ldr];  // Row ii of R (diagonal element of V^T)

    // Initialize W[ii,:] = R[ii,:] (the diagonal 1 in V^T)
    size_t j = 0;
    for (; j + MAT_NEON_WIDTH * 4 <= N; j += MAT_NEON_WIDTH * 4) {
      MAT_NEON_STORE(&Wi[j], MAT_NEON_LOAD(&Rii[j]));
      MAT_NEON_STORE(&Wi[j + MAT_NEON_WIDTH], MAT_NEON_LOAD(&Rii[j + MAT_NEON_WIDTH]));
      MAT_NEON_STORE(&Wi[j + MAT_NEON_WIDTH * 2], MAT_NEON_LOAD(&Rii[j + MAT_NEON_WIDTH * 2]));
      MAT_NEON_STORE(&Wi[j + MAT_NEON_WIDTH * 3], MAT_NEON_LOAD(&Rii[j + MAT_NEON_WIDTH * 3]));
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
MAT_INTERNAL_STATIC void mat_gemm_unit_lower_neon(
    mat_elem_t *C, size_t ldc,
    const mat_elem_t *V, size_t ldv,
    const mat_elem_t *W, size_t ldw,
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

// W = Q * V where Q is (M x panel_rows) with stride ldq, V is unit lower (panel_rows x kb)
// Result W is (M x kb) with stride ldw
MAT_INTERNAL_STATIC void mat_gemm_q_unit_lower_neon(
    mat_elem_t *W, size_t ldw,
    const mat_elem_t *Q, size_t ldq,
    const mat_elem_t *V, size_t ldv,
    size_t M, size_t panel_rows, size_t kb) {

  for (size_t ii = 0; ii < M; ii++) {
    const mat_elem_t *Qi = &Q[ii * ldq];
    mat_elem_t *Wi = &W[ii * ldw];

    for (size_t jj = 0; jj < kb; jj++) {
      // W[ii, jj] = Q[ii, jj] * 1 + sum(Q[ii, r] * V[jj, r]) for r > jj
      mat_elem_t dot = Qi[jj];  // diagonal 1
      for (size_t r = jj + 1; r < panel_rows; r++) {
        dot += Qi[r] * V[jj * ldv + r];
      }
      Wi[jj] = dot;
    }
  }
}

// Q -= W * V^T where W is (M x kb), V^T is unit upper (kb x panel_rows)
// Q is (M x panel_rows) with stride ldq
MAT_INTERNAL_STATIC void mat_gemm_sub_w_unit_upper_neon(
    mat_elem_t *Q, size_t ldq,
    const mat_elem_t *W, size_t ldw,
    const mat_elem_t *V, size_t ldv,
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
MAT_INTERNAL_STATIC void mat_gemm_unit_lower_t_scalar(
    mat_elem_t *W, size_t ldw,
    const mat_elem_t *V, size_t ldv,
    const mat_elem_t *R, size_t ldr,
    size_t kb, size_t panel_rows, size_t N) {
  for (size_t ii = 0; ii < kb; ii++) {
    for (size_t jj = 0; jj < N; jj++) {
      mat_elem_t dot = R[ii * ldr + jj];  // diagonal 1
      for (size_t r = ii + 1; r < panel_rows; r++)
        dot += V[ii * ldv + r] * R[r * ldr + jj];
      W[ii * ldw + jj] = dot;
    }
  }
}

MAT_INTERNAL_STATIC void mat_gemm_unit_lower_scalar(
    mat_elem_t *C, size_t ldc,
    const mat_elem_t *V, size_t ldv,
    const mat_elem_t *W, size_t ldw,
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

MAT_INTERNAL_STATIC void mat_gemm_q_unit_lower_scalar(
    mat_elem_t *W, size_t ldw,
    const mat_elem_t *Q, size_t ldq,
    const mat_elem_t *V, size_t ldv,
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

MAT_INTERNAL_STATIC void mat_gemm_sub_w_unit_upper_scalar(
    mat_elem_t *Q, size_t ldq,
    const mat_elem_t *W, size_t ldw,
    const mat_elem_t *V, size_t ldv,
    size_t M, size_t kb, size_t panel_rows) {
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

// Strided GEMM: C = alpha * A * B + beta * C
// For submatrix operations where matrices have leading dimensions != their logical width
// A is M x K with stride lda, B is K x N with stride ldb, C is M x N with stride ldc
#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void mat_gemm_strided_neon_impl(
    mat_elem_t *C, size_t ldc,
    mat_elem_t alpha,
    const mat_elem_t *A, size_t lda, size_t M, size_t K,
    const mat_elem_t *B, size_t ldb, size_t N,
    mat_elem_t beta) {

  // Scale C by beta first
  if (beta == 0) {
    for (size_t i = 0; i < M; i++)
      memset(&C[i * ldc], 0, N * sizeof(mat_elem_t));
  } else if (beta != 1) {
    for (size_t i = 0; i < M; i++) {
      mat_elem_t *Ci = &C[i * ldc];
      for (size_t j = 0; j < N; j++)
        Ci[j] *= beta;
    }
  }

  // Transpose B for cache-friendly access
  mat_elem_t *bt = (mat_elem_t *)mat_scratch_alloc_(K * N * sizeof(mat_elem_t));
  for (size_t j = 0; j < N; j++) {
    for (size_t k = 0; k < K; k++) {
      bt[j * K + k] = B[k * ldb + j];
    }
  }

  // 4x4 micro-kernel with strided access
  size_t i = 0;
  for (; i + 4 <= M; i += 4) {
    size_t j = 0;
    for (; j + 4 <= N; j += 4) {
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
        MAT_NEON_TYPE a0 = MAT_NEON_LOAD(&A[(i+0)*lda + k]);
        MAT_NEON_TYPE a1 = MAT_NEON_LOAD(&A[(i+1)*lda + k]);
        MAT_NEON_TYPE a2 = MAT_NEON_LOAD(&A[(i+2)*lda + k]);
        MAT_NEON_TYPE a3 = MAT_NEON_LOAD(&A[(i+3)*lda + k]);
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

      // Horizontal sum and add to C with alpha scaling
      C[(i+0)*ldc+(j+0)] += alpha * MAT_NEON_ADDV(acc00);
      C[(i+0)*ldc+(j+1)] += alpha * MAT_NEON_ADDV(acc01);
      C[(i+0)*ldc+(j+2)] += alpha * MAT_NEON_ADDV(acc02);
      C[(i+0)*ldc+(j+3)] += alpha * MAT_NEON_ADDV(acc03);
      C[(i+1)*ldc+(j+0)] += alpha * MAT_NEON_ADDV(acc10);
      C[(i+1)*ldc+(j+1)] += alpha * MAT_NEON_ADDV(acc11);
      C[(i+1)*ldc+(j+2)] += alpha * MAT_NEON_ADDV(acc12);
      C[(i+1)*ldc+(j+3)] += alpha * MAT_NEON_ADDV(acc13);
      C[(i+2)*ldc+(j+0)] += alpha * MAT_NEON_ADDV(acc20);
      C[(i+2)*ldc+(j+1)] += alpha * MAT_NEON_ADDV(acc21);
      C[(i+2)*ldc+(j+2)] += alpha * MAT_NEON_ADDV(acc22);
      C[(i+2)*ldc+(j+3)] += alpha * MAT_NEON_ADDV(acc23);
      C[(i+3)*ldc+(j+0)] += alpha * MAT_NEON_ADDV(acc30);
      C[(i+3)*ldc+(j+1)] += alpha * MAT_NEON_ADDV(acc31);
      C[(i+3)*ldc+(j+2)] += alpha * MAT_NEON_ADDV(acc32);
      C[(i+3)*ldc+(j+3)] += alpha * MAT_NEON_ADDV(acc33);

      // Scalar remainder k
      for (; k < K; k++) {
        mat_elem_t a0s = A[(i+0)*lda + k], a1s = A[(i+1)*lda + k];
        mat_elem_t a2s = A[(i+2)*lda + k], a3s = A[(i+3)*lda + k];
        for (size_t jj = 0; jj < 4; jj++) {
          mat_elem_t bval = bt[(j+jj)*K + k];
          C[(i+0)*ldc+(j+jj)] += alpha * a0s * bval;
          C[(i+1)*ldc+(j+jj)] += alpha * a1s * bval;
          C[(i+2)*ldc+(j+jj)] += alpha * a2s * bval;
          C[(i+3)*ldc+(j+jj)] += alpha * a3s * bval;
        }
      }
    }

    // Remainder j columns
    for (; j < N; j++) {
      for (size_t ii = 0; ii < 4; ii++) {
        mat_elem_t sum = 0;
        for (size_t k = 0; k < K; k++)
          sum += A[(i+ii)*lda + k] * bt[j*K + k];
        C[(i+ii)*ldc + j] += alpha * sum;
      }
    }
  }

  // Remainder i rows
  for (; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      mat_elem_t sum = 0;
      for (size_t k = 0; k < K; k++)
        sum += A[i*lda + k] * bt[j*K + k];
      C[i*ldc + j] += alpha * sum;
    }
  }

#ifndef MAT_NO_SCRATCH
  mat_scratch_reset_();
#else
  mat_scratch_free_(bt);
#endif
}
#endif

MAT_INTERNAL_STATIC void mat_gemm_strided_scalar_impl(
    mat_elem_t *C, size_t ldc,
    mat_elem_t alpha,
    const mat_elem_t *A, size_t lda, size_t M, size_t K,
    const mat_elem_t *B, size_t ldb, size_t N,
    mat_elem_t beta) {

  // Scale C by beta first
  for (size_t i = 0; i < M; i++) {
    mat_elem_t *Ci = &C[i * ldc];
    for (size_t j = 0; j < N; j++)
      Ci[j] *= beta;
  }

  // ikj loop order for cache-friendly access
  for (size_t i = 0; i < M; i++) {
    for (size_t k = 0; k < K; k++) {
      mat_elem_t aik = alpha * A[i * lda + k];
      mat_elem_t *Ci = &C[i * ldc];
      const mat_elem_t *Bk = &B[k * ldb];
      for (size_t j = 0; j < N; j++)
        Ci[j] += aik * Bk[j];
    }
  }
}

MAT_INTERNAL_STATIC void mat_gemm_strided(
    mat_elem_t *C, size_t ldc,
    mat_elem_t alpha,
    const mat_elem_t *A, size_t lda, size_t M, size_t K,
    const mat_elem_t *B, size_t ldb, size_t N,
    mat_elem_t beta) {
#ifdef MAT_HAS_ARM_NEON
  mat_gemm_strided_neon_impl(C, ldc, alpha, A, lda, M, K, B, ldb, N, beta);
#else
  mat_gemm_strided_scalar_impl(C, ldc, alpha, A, lda, M, K, B, ldb, N, beta);
#endif
}

// C = alpha * A * B + beta * C (BLAS Level-3: gemm)
#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void mat_gemm_neon_impl(Mat *C, mat_elem_t alpha, const Mat *A, const Mat *B, mat_elem_t beta) {
  size_t M = A->rows;
  size_t K = A->cols;
  size_t N = B->cols;

  // Scale C by beta first
  if (beta == 0) {
    memset(C->data, 0, M * N * sizeof(mat_elem_t));
  } else if (beta != 1) {
    size_t len = M * N;
    MAT_NEON_TYPE vbeta = MAT_NEON_DUP(beta);
    size_t i = 0;
    for (; i + MAT_NEON_WIDTH <= len; i += MAT_NEON_WIDTH) {
      MAT_NEON_TYPE vc = MAT_NEON_LOAD(&C->data[i]);
      MAT_NEON_STORE(&C->data[i], MAT_NEON_FMA(MAT_NEON_DUP(0), vc, vbeta));
    }
    for (; i < len; i++) {
      C->data[i] *= beta;
    }
  }

  // Transpose B for cache-friendly access (using scratch arena)
  mat_elem_t *bt = (mat_elem_t *)mat_scratch_alloc_(K * N * sizeof(mat_elem_t));
  Mat Bt_storage = { .rows = N, .cols = K, .data = bt };
  mat_t(&Bt_storage, B);

  // 4x4 micro-kernel with OpenMP parallelization on outer loop
  size_t M4 = (M / 4) * 4;  // Round down to multiple of 4
  size_t N4 = (N / 4) * 4;

#ifdef MAT_HAS_OPENMP
  #pragma omp parallel for schedule(static) if(M * N * K >= MAT_OMP_THRESHOLD)
#endif
  for (size_t i = 0; i < M4; i += 4) {
    for (size_t j = 0; j < N4; j += 4) {
      MAT_NEON_TYPE acc00 = MAT_NEON_DUP(0), acc01 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc02 = MAT_NEON_DUP(0), acc03 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc10 = MAT_NEON_DUP(0), acc11 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc12 = MAT_NEON_DUP(0), acc13 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc20 = MAT_NEON_DUP(0), acc21 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc22 = MAT_NEON_DUP(0), acc23 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc30 = MAT_NEON_DUP(0), acc31 = MAT_NEON_DUP(0);
      MAT_NEON_TYPE acc32 = MAT_NEON_DUP(0), acc33 = MAT_NEON_DUP(0);

      size_t k = 0;
      // 4x unrolled inner loop for better instruction-level parallelism
      for (; k + MAT_NEON_WIDTH * 4 <= K; k += MAT_NEON_WIDTH * 4) {
        #define GEMM_KERNEL_STEP(off) do { \
          MAT_NEON_TYPE a0 = MAT_NEON_LOAD(&A->data[(i+0)*K + k + (off)]); \
          MAT_NEON_TYPE a1 = MAT_NEON_LOAD(&A->data[(i+1)*K + k + (off)]); \
          MAT_NEON_TYPE a2 = MAT_NEON_LOAD(&A->data[(i+2)*K + k + (off)]); \
          MAT_NEON_TYPE a3 = MAT_NEON_LOAD(&A->data[(i+3)*K + k + (off)]); \
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
        GEMM_KERNEL_STEP(0);
        GEMM_KERNEL_STEP(MAT_NEON_WIDTH);
        GEMM_KERNEL_STEP(MAT_NEON_WIDTH * 2);
        GEMM_KERNEL_STEP(MAT_NEON_WIDTH * 3);
        #undef GEMM_KERNEL_STEP
      }

      // Handle remaining k in single NEON-width steps
      for (; k + MAT_NEON_WIDTH <= K; k += MAT_NEON_WIDTH) {
        MAT_NEON_TYPE a0 = MAT_NEON_LOAD(&A->data[(i+0)*K + k]);
        MAT_NEON_TYPE a1 = MAT_NEON_LOAD(&A->data[(i+1)*K + k]);
        MAT_NEON_TYPE a2 = MAT_NEON_LOAD(&A->data[(i+2)*K + k]);
        MAT_NEON_TYPE a3 = MAT_NEON_LOAD(&A->data[(i+3)*K + k]);
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

      // Horizontal sum and add to C with alpha scaling
      C->data[(i+0)*N+(j+0)] += alpha * MAT_NEON_ADDV(acc00);
      C->data[(i+0)*N+(j+1)] += alpha * MAT_NEON_ADDV(acc01);
      C->data[(i+0)*N+(j+2)] += alpha * MAT_NEON_ADDV(acc02);
      C->data[(i+0)*N+(j+3)] += alpha * MAT_NEON_ADDV(acc03);
      C->data[(i+1)*N+(j+0)] += alpha * MAT_NEON_ADDV(acc10);
      C->data[(i+1)*N+(j+1)] += alpha * MAT_NEON_ADDV(acc11);
      C->data[(i+1)*N+(j+2)] += alpha * MAT_NEON_ADDV(acc12);
      C->data[(i+1)*N+(j+3)] += alpha * MAT_NEON_ADDV(acc13);
      C->data[(i+2)*N+(j+0)] += alpha * MAT_NEON_ADDV(acc20);
      C->data[(i+2)*N+(j+1)] += alpha * MAT_NEON_ADDV(acc21);
      C->data[(i+2)*N+(j+2)] += alpha * MAT_NEON_ADDV(acc22);
      C->data[(i+2)*N+(j+3)] += alpha * MAT_NEON_ADDV(acc23);
      C->data[(i+3)*N+(j+0)] += alpha * MAT_NEON_ADDV(acc30);
      C->data[(i+3)*N+(j+1)] += alpha * MAT_NEON_ADDV(acc31);
      C->data[(i+3)*N+(j+2)] += alpha * MAT_NEON_ADDV(acc32);
      C->data[(i+3)*N+(j+3)] += alpha * MAT_NEON_ADDV(acc33);

      // Scalar remainder k
      for (; k < K; k++) {
        mat_elem_t a0s = A->data[(i+0)*K + k], a1s = A->data[(i+1)*K + k];
        mat_elem_t a2s = A->data[(i+2)*K + k], a3s = A->data[(i+3)*K + k];
        for (size_t jj = 0; jj < 4; jj++) {
          mat_elem_t bval = bt[(j+jj)*K + k];
          C->data[(i+0)*N+(j+jj)] += alpha * a0s * bval;
          C->data[(i+1)*N+(j+jj)] += alpha * a1s * bval;
          C->data[(i+2)*N+(j+jj)] += alpha * a2s * bval;
          C->data[(i+3)*N+(j+jj)] += alpha * a3s * bval;
        }
      }
    }

    // Remainder j columns (within this row block)
    for (size_t jj = N4; jj < N; jj++) {
      for (size_t ii = 0; ii < 4; ii++) {
        mat_elem_t sum = 0;
        for (size_t k = 0; k < K; k++) {
          sum += A->data[(i+ii)*K + k] * bt[jj*K + k];
        }
        C->data[(i+ii)*N + jj] += alpha * sum;
      }
    }
  }

  // Remainder i rows (after parallel region)
  for (size_t i = M4; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      mat_elem_t sum = 0;
      for (size_t k = 0; k < K; k++) {
        sum += A->data[i*K + k] * bt[j*K + k];
      }
      C->data[i*N + j] += alpha * sum;
    }
  }

#ifndef MAT_NO_SCRATCH
  mat_scratch_reset_();
#else
  mat_scratch_free_(bt);
#endif
}
#endif

MAT_INTERNAL_STATIC void mat_gemm_scalar_impl(Mat *C, mat_elem_t alpha, const Mat *A, const Mat *B, mat_elem_t beta) {
  size_t M = A->rows;
  size_t K = A->cols;
  size_t N = B->cols;

  // Scale C by beta first, then accumulate
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      C->data[i * N + j] *= beta;
    }
  }

  // ikj loop order for cache-friendly access
#ifdef MAT_HAS_OPENMP
  #pragma omp parallel for schedule(static) if(M * N * K >= MAT_OMP_THRESHOLD)
#endif
  for (size_t i = 0; i < M; i++) {
    for (size_t k = 0; k < K; k++) {
      mat_elem_t aik = alpha * A->data[i * K + k];
      for (size_t j = 0; j < N; j++) {
        C->data[i * N + j] += aik * B->data[k * N + j];
      }
    }
  }
}

MATDEF void mat_gemm(Mat *C, mat_elem_t alpha, const Mat *A, const Mat *B, mat_elem_t beta) {
  MAT_ASSERT_MAT(C);
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(B);
  MAT_ASSERT(A->cols == B->rows);
  MAT_ASSERT(C->rows == A->rows && C->cols == B->cols);

#ifdef MAT_HAS_ARM_NEON
  mat_gemm_neon_impl(C, alpha, A, B, beta);
#else
  mat_gemm_scalar_impl(C, alpha, A, B, beta);
#endif
}

/* Structure Operations */

#define MAT_T_BLOCK 32

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC void mat_t_neon_impl(Mat *out, const Mat *m) {
  size_t rows = m->rows;
  size_t cols = m->cols;
  const mat_elem_t *src = m->data;
  mat_elem_t *dst = out->data;

  size_t full_rows = (rows / MAT_T_BLOCK) * MAT_T_BLOCK;
  size_t full_cols = (cols / MAT_T_BLOCK) * MAT_T_BLOCK;

  // Main blocked loop - parallelizable
#if defined(MAT_HAS_OPENMP)
  #pragma omp parallel for collapse(2) schedule(static) if(rows * cols >= MAT_OMP_THRESHOLD)
#endif
  for (size_t ii = 0; ii < full_rows; ii += MAT_T_BLOCK) {
    for (size_t jj = 0; jj < full_cols; jj += MAT_T_BLOCK) {
#ifdef MAT_DOUBLE_PRECISION
      // float64x2_t holds 2 doubles, process 2x2 blocks
      for (size_t i = ii; i < ii + MAT_T_BLOCK; i += 2) {
        for (size_t j = jj; j < jj + MAT_T_BLOCK; j += 2) {
          float64x2_t r0 = vld1q_f64(&src[(i+0) * cols + j]);
          float64x2_t r1 = vld1q_f64(&src[(i+1) * cols + j]);
          float64x2_t t0 = vzip1q_f64(r0, r1);
          float64x2_t t1 = vzip2q_f64(r0, r1);
          vst1q_f64(&dst[(j+0) * rows + i], t0);
          vst1q_f64(&dst[(j+1) * rows + i], t1);
        }
      }
#else
      // float32x4_t holds 4 floats, process 4x4 blocks
      for (size_t i = ii; i < ii + MAT_T_BLOCK; i += 4) {
        for (size_t j = jj; j < jj + MAT_T_BLOCK; j += 4) {
          float32x4_t r0 = vld1q_f32(&src[(i+0) * cols + j]);
          float32x4_t r1 = vld1q_f32(&src[(i+1) * cols + j]);
          float32x4_t r2 = vld1q_f32(&src[(i+2) * cols + j]);
          float32x4_t r3 = vld1q_f32(&src[(i+3) * cols + j]);
          float32x4x2_t p01 = vtrnq_f32(r0, r1);
          float32x4x2_t p23 = vtrnq_f32(r2, r3);
          float32x4_t t0 = vcombine_f32(vget_low_f32(p01.val[0]), vget_low_f32(p23.val[0]));
          float32x4_t t1 = vcombine_f32(vget_low_f32(p01.val[1]), vget_low_f32(p23.val[1]));
          float32x4_t t2 = vcombine_f32(vget_high_f32(p01.val[0]), vget_high_f32(p23.val[0]));
          float32x4_t t3 = vcombine_f32(vget_high_f32(p01.val[1]), vget_high_f32(p23.val[1]));
          vst1q_f32(&dst[(j+0) * rows + i], t0);
          vst1q_f32(&dst[(j+1) * rows + i], t1);
          vst1q_f32(&dst[(j+2) * rows + i], t2);
          vst1q_f32(&dst[(j+3) * rows + i], t3);
        }
      }
#endif
    }
  }

  // Edge cases: right edge (cols not multiple of block)
  for (size_t ii = 0; ii < full_rows; ii += MAT_T_BLOCK) {
    for (size_t i = ii; i < ii + MAT_T_BLOCK; i++) {
      for (size_t j = full_cols; j < cols; j++) {
        dst[j * rows + i] = src[i * cols + j];
      }
    }
  }

  // Edge cases: bottom edge (rows not multiple of block)
  for (size_t i = full_rows; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      dst[j * rows + i] = src[i * cols + j];
    }
  }
}
#endif

MAT_INTERNAL_STATIC void mat_t_scalar_impl(Mat *out, const Mat *m) {
  size_t rows = m->rows;
  size_t cols = m->cols;
  const mat_elem_t *src = m->data;
  mat_elem_t *dst = out->data;

#ifdef MAT_HAS_OPENMP
  #pragma omp parallel for collapse(2) schedule(static) if(rows * cols >= MAT_OMP_THRESHOLD)
#endif
  for (size_t ii = 0; ii < rows; ii += MAT_T_BLOCK) {
    for (size_t jj = 0; jj < cols; jj += MAT_T_BLOCK) {
      size_t i_end = (ii + MAT_T_BLOCK < rows) ? ii + MAT_T_BLOCK : rows;
      size_t j_end = (jj + MAT_T_BLOCK < cols) ? jj + MAT_T_BLOCK : cols;
      for (size_t i = ii; i < i_end; i++) {
        for (size_t j = jj; j < j_end; j++) {
          dst[j * rows + i] = src[i * cols + j];
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
  mat_t_neon_impl(out, m);
#else
  mat_t_scalar_impl(out, m);
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

MATDEF Mat *mat_slice(const Mat *m, size_t row_start, size_t row_end, size_t col_start, size_t col_end) {
  MAT_ASSERT_MAT(m);
  MAT_ASSERT(row_start < row_end && row_end <= m->rows);
  MAT_ASSERT(col_start < col_end && col_end <= m->cols);

  size_t out_rows = row_end - row_start;
  size_t out_cols = col_end - col_start;
  Mat *result = mat_mat(out_rows, out_cols);

  for (size_t i = 0; i < out_rows; i++) {
    memcpy(&result->data[i * out_cols],
           &m->data[(row_start + i) * m->cols + col_start],
           out_cols * sizeof(mat_elem_t));
  }

  return result;
}

MATDEF void mat_slice_set(Mat *m, size_t row_start, size_t col_start, const Mat *src) {
  MAT_ASSERT_MAT(m);
  MAT_ASSERT_MAT(src);
  MAT_ASSERT(row_start + src->rows <= m->rows);
  MAT_ASSERT(col_start + src->cols <= m->cols);

  for (size_t i = 0; i < src->rows; i++) {
    memcpy(&m->data[(row_start + i) * m->cols + col_start],
           &src->data[i * src->cols],
           src->cols * sizeof(mat_elem_t));
  }
}

MATDEF void mat_hcat(Mat *out, const Mat *a, const Mat *b) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);
  MAT_ASSERT_MAT(b);
  MAT_ASSERT(a->rows == b->rows && "mat_hcat: row count must match");
  MAT_ASSERT(out->rows == a->rows && out->cols == a->cols + b->cols);

  for (size_t i = 0; i < a->rows; i++) {
    memcpy(&out->data[i * out->cols], &a->data[i * a->cols], a->cols * sizeof(mat_elem_t));
    memcpy(&out->data[i * out->cols + a->cols], &b->data[i * b->cols], b->cols * sizeof(mat_elem_t));
  }
}

MATDEF void mat_vcat(Mat *out, const Mat *a, const Mat *b) {
  MAT_ASSERT_MAT(out);
  MAT_ASSERT_MAT(a);
  MAT_ASSERT_MAT(b);
  MAT_ASSERT(a->cols == b->cols && "mat_vcat: column count must match");
  MAT_ASSERT(out->rows == a->rows + b->rows && out->cols == a->cols);

  memcpy(out->data, a->data, a->rows * a->cols * sizeof(mat_elem_t));
  memcpy(&out->data[a->rows * a->cols], b->data, b->rows * b->cols * sizeof(mat_elem_t));
}

MATDEF Vec *mat_row(const Mat *m, size_t row) {
  MAT_ASSERT_MAT(m);
  MAT_ASSERT(row < m->rows && "mat_row: row index out of bounds");

  Vec *v = mat_vec(m->cols);
  memcpy(v->data, &m->data[row * m->cols], m->cols * sizeof(mat_elem_t));
  return v;
}

MATDEF Vec *mat_col(const Mat *m, size_t col) {
  MAT_ASSERT_MAT(m);
  MAT_ASSERT(col < m->cols && "mat_col: column index out of bounds");

  Vec *v = mat_vec(m->rows);
  for (size_t i = 0; i < m->rows; i++) {
    v->data[i] = m->data[i * m->cols + col];
  }
  return v;
}

/* Diagonal Operations */

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
  return mat_norm_fro(a);
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

MAT_INTERNAL_STATIC mat_elem_t mat_norm_fro_scalar_impl(const Mat *a) {
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
  return mat_norm_fro_neon_impl(a);
#else
  return mat_norm_fro_scalar_impl(a);
#endif
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC mat_elem_t mat_norm_fro_fast_neon_impl(const Mat *a) {
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
  return mat_norm_fro_fast_neon_impl(a);
#else
  return mat_norm_fro_scalar_impl(a);
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

/* Reduction Operations */

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC mat_elem_t mat_sum_neon_impl(const Mat *a) {
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

MAT_INTERNAL_STATIC mat_elem_t mat_sum_scalar_impl(const Mat *a) {
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
  return mat_sum_neon_impl(a);
#else
  return mat_sum_scalar_impl(a);
#endif
}

MATDEF mat_elem_t mat_mean(const Mat *a) {
  MAT_ASSERT_MAT(a);
  return mat_sum(a) / (mat_elem_t)(a->rows * a->cols);
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC mat_elem_t mat_min_neon_impl(const Mat *a) {
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
    if (pa[i] < min_val) min_val = pa[i];
  }

  return min_val;
}
#endif

MAT_INTERNAL_STATIC mat_elem_t mat_min_scalar_impl(const Mat *a) {
  size_t len = a->rows * a->cols;
  mat_elem_t *pa = a->data;

  mat_elem_t min_val = pa[0];

  for (size_t i = 1; i < len; i++) {
    if (pa[i] < min_val) min_val = pa[i];
  }

  return min_val;
}

MATDEF mat_elem_t mat_min(const Mat *a) {
  MAT_ASSERT_MAT(a);

#ifdef MAT_HAS_ARM_NEON
  return mat_min_neon_impl(a);
#else
  return mat_min_scalar_impl(a);
#endif
}

#ifdef MAT_HAS_ARM_NEON
MAT_INTERNAL_STATIC mat_elem_t mat_max_neon_impl(const Mat *a) {
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
    if (pa[i] > max_val) max_val = pa[i];
  }

  return max_val;
}
#endif

MAT_INTERNAL_STATIC mat_elem_t mat_max_scalar_impl(const Mat *a) {
  size_t len = a->rows * a->cols;
  mat_elem_t *pa = a->data;

  mat_elem_t max_val = pa[0];

  for (size_t i = 1; i < len; i++) {
    if (pa[i] > max_val) max_val = pa[i];
  }

  return max_val;
}

MATDEF mat_elem_t mat_max(const Mat *a) {
  MAT_ASSERT_MAT(a);

#ifdef MAT_HAS_ARM_NEON
  return mat_max_neon_impl(a);
#else
  return mat_max_scalar_impl(a);
#endif
}

MATDEF void mat_sum_axis(Vec *out, const Mat *a, int axis) {
  MAT_ASSERT_MAT(a);
  MAT_ASSERT_MAT(out);

  if (axis == 0) {
    // Sum along columns: result has shape (rows, 1) = (a->rows,)
    MAT_ASSERT(out->rows * out->cols == a->rows && "mat_sum_axis: output size must match rows");

    for (size_t i = 0; i < a->rows; i++) {
      mat_elem_t sum = 0;
      for (size_t j = 0; j < a->cols; j++) {
        sum += a->data[i * a->cols + j];
      }
      out->data[i] = sum;
    }
  } else {
    // Sum along rows (axis=1): result has shape (1, cols) = (a->cols,)
    MAT_ASSERT(out->rows * out->cols == a->cols && "mat_sum_axis: output size must match cols");

    // Zero output
    for (size_t j = 0; j < a->cols; j++) {
      out->data[j] = 0;
    }

    for (size_t i = 0; i < a->rows; i++) {
      for (size_t j = 0; j < a->cols; j++) {
        out->data[j] += a->data[i * a->cols + j];
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

// QR Decomposition using blocked Householder (WY representation)
// A (m x n) = Q (m x m) * R (m x n), requires m >= n
#define MAT_QR_BLOCK_SIZE 32

MATDEF void mat_qr(const Mat *A, Mat *Q, Mat *R) {
  MAT_ASSERT_MAT(A);
  MAT_ASSERT_MAT(Q);
  MAT_ASSERT_MAT(R);

  size_t m = A->rows;
  size_t n = A->cols;
  MAT_ASSERT(m >= n && "mat_qr requires m >= n");
  MAT_ASSERT(Q->rows == m && Q->cols == m);
  MAT_ASSERT(R->rows == m && R->cols == n);

  // Copy A to R (we transform R in-place)
  mat_deep_copy(R, A);

  // Initialize Q to identity
  mat_eye(Q);

  size_t nb = MAT_QR_BLOCK_SIZE;
  if (nb > n) nb = n;

  // Allocate workspace:
  // V: m x nb - Householder vectors for current block
  // T: nb x nb - upper triangular matrix for WY representation
  // W: workspace for applying block reflector (max(m, n) x nb)
  mat_elem_t *V = (mat_elem_t *)MAT_MALLOC(m * nb * sizeof(mat_elem_t));
  mat_elem_t *T = (mat_elem_t *)MAT_MALLOC(nb * nb * sizeof(mat_elem_t));
  mat_elem_t *W = (mat_elem_t *)MAT_MALLOC(m * nb * sizeof(mat_elem_t));
  mat_elem_t *tau_arr = (mat_elem_t *)MAT_MALLOC(nb * sizeof(mat_elem_t));
  MAT_ASSERT(V && T && W && tau_arr);

  for (size_t k = 0; k < n; k += nb) {
    size_t kb = (k + nb <= n) ? nb : (n - k);  // actual block size
    size_t panel_rows = m - k;

    // === Panel factorization: factor columns k to k+kb ===
    // Zero out V and T for this block
    for (size_t i = 0; i < panel_rows * kb; i++) V[i] = 0;
    for (size_t i = 0; i < kb * kb; i++) T[i] = 0;

    for (size_t j = 0; j < kb; j++) {
      size_t col = k + j;
      size_t len = m - col;

      // Compute norm of R[col:m, col]
      mat_elem_t norm_x = 0;
      for (size_t i = 0; i < len; i++) {
        mat_elem_t val = R->data[(col + i) * n + col];
        norm_x += val * val;
      }
#ifdef MAT_DOUBLE_PRECISION
      norm_x = sqrt(norm_x);
#else
      norm_x = sqrtf(norm_x);
#endif

      mat_elem_t tau = 0;
      if (norm_x >= MAT_DEFAULT_EPSILON) {
        // Form Householder vector
        mat_elem_t x0 = R->data[col * n + col];
        mat_elem_t sign = (x0 >= 0) ? 1 : -1;
        mat_elem_t v0 = x0 + sign * norm_x;

        // Store normalized v in V (v[0] = 1, rest scaled)
        V[j * panel_rows + j] = 1;  // v[0] = 1 after normalization
        for (size_t i = 1; i < len; i++) {
          V[j * panel_rows + j + i] = R->data[(col + i) * n + col] / v0;
        }

        // tau = 2 / (v^T v) where v[0] = 1
        mat_elem_t vtv = 1;
        for (size_t i = 1; i < len; i++) {
          mat_elem_t vi = V[j * panel_rows + j + i];
          vtv += vi * vi;
        }
        tau = 2 / vtv;

        // Apply H to R[col:m, col:k+kb] (panel columns)
        for (size_t jj = col; jj < k + kb; jj++) {
          mat_elem_t dot = R->data[col * n + jj];  // v[0] = 1
          for (size_t i = 1; i < len; i++) {
            dot += V[j * panel_rows + j + i] * R->data[(col + i) * n + jj];
          }
          dot *= tau;
          R->data[col * n + jj] -= dot;  // v[0] = 1
          for (size_t i = 1; i < len; i++) {
            R->data[(col + i) * n + jj] -= dot * V[j * panel_rows + j + i];
          }
        }
      }
      tau_arr[j] = tau;
    }

    // === Build T matrix (upper triangular for forward product H_0 * H_1 * ... * H_{kb-1}) ===
    // T(j, j) = tau(j)
    // T(0:j-1, j) = -tau(j) * T(0:j-1, 0:j-1) * V(:, 0:j-1)^T * V(:, j)
    for (size_t j = 0; j < kb; j++) {
      T[j * kb + j] = tau_arr[j];
      if (tau_arr[j] == 0 || j == 0) continue;

      // Compute z[i] = V(:, i)^T * V(:, j) for i = 0, ..., j-1
      // Store in W temporarily
      for (size_t i = 0; i < j; i++) {
        // V(:,i) has 1 at position i, then V[i*panel_rows + r] for r > i
        // V(:,j) has 1 at position j, then V[j*panel_rows + r] for r > j
        // Since j > i, the overlap starts at position j
        mat_elem_t dot = V[i * panel_rows + j];  // V(:,i)[j] * 1
        for (size_t r = j + 1; r < panel_rows; r++) {
          dot += V[i * panel_rows + r] * V[j * panel_rows + r];
        }
        W[i] = dot;
      }

      // T(0:j-1, j) = -tau(j) * T(0:j-1, 0:j-1) * z
      // Upper triangular matrix-vector multiply
      for (size_t i = 0; i < j; i++) {
        mat_elem_t sum = 0;
        for (size_t p = i; p < j; p++) {
          sum += T[i * kb + p] * W[p];
        }
        T[i * kb + j] = -tau_arr[j] * sum;
      }
    }

    // === Apply block reflector to trailing matrix R[k:m, k+kb:n] ===
    if (k + kb < n) {
      size_t trail_cols = n - k - kb;

      // W = V^T * R[k:m, k+kb:n]  (kb x trail_cols)
#ifdef MAT_HAS_ARM_NEON
      mat_gemm_unit_lower_t_neon(W, trail_cols, V, panel_rows,
                                 &R->data[k * n + (k + kb)], n,
                                 kb, panel_rows, trail_cols);
#else
      mat_gemm_unit_lower_t_scalar(W, trail_cols, V, panel_rows,
                                   &R->data[k * n + (k + kb)], n,
                                   kb, panel_rows, trail_cols);
#endif

      // W = T^T * W  (kb x trail_cols), using T^T (lower triangular) for H_k*...*H_1
      for (size_t jj = 0; jj < trail_cols; jj++) {
        for (int ii = (int)kb - 1; ii >= 0; ii--) {
          mat_elem_t sum = 0;
          for (size_t p = 0; p <= (size_t)ii; p++) {
            sum += T[p * kb + ii] * W[p * trail_cols + jj];
          }
          W[ii * trail_cols + jj] = sum;
        }
      }

      // R[k:m, k+kb:n] -= V * W
#ifdef MAT_HAS_ARM_NEON
      mat_gemm_unit_lower_neon(&R->data[k * n + (k + kb)], n,
                               V, panel_rows, W, trail_cols,
                               panel_rows, kb, trail_cols);
#else
      mat_gemm_unit_lower_scalar(&R->data[k * n + (k + kb)], n,
                                 V, panel_rows, W, trail_cols,
                                 panel_rows, kb, trail_cols);
#endif
    }

    // === Apply block reflector to Q[:, k:m] ===
    // Q = Q * (I - V * T * V^T) = Q - Q * V * T * V^T
    // Let W = Q[:, k:m] * V  (m x kb)
#ifdef MAT_HAS_ARM_NEON
    mat_gemm_q_unit_lower_neon(W, kb, &Q->data[k], m, V, panel_rows,
                               m, panel_rows, kb);
#else
    mat_gemm_q_unit_lower_scalar(W, kb, &Q->data[k], m, V, panel_rows,
                                 m, panel_rows, kb);
#endif

    // W = W * T  (m x kb), T is upper triangular
    for (size_t ii = 0; ii < m; ii++) {
      for (int jj = (int)kb - 1; jj >= 0; jj--) {  // bottom to top for in-place
        mat_elem_t sum = 0;
        for (size_t p = 0; p <= (size_t)jj; p++) {
          sum += W[ii * kb + p] * T[p * kb + jj];
        }
        W[ii * kb + jj] = sum;
      }
    }

    // Q[:, k:m] -= W * V^T
#ifdef MAT_HAS_ARM_NEON
    mat_gemm_sub_w_unit_upper_neon(&Q->data[k], m, W, kb, V, panel_rows,
                                   m, kb, panel_rows);
#else
    mat_gemm_sub_w_unit_upper_scalar(&Q->data[k], m, W, kb, V, panel_rows,
                                     m, kb, panel_rows);
#endif
  }

  // Zero out lower triangular part of R
  for (size_t i = 1; i < m; i++) {
    for (size_t j = 0; j < i && j < n; j++) {
      R->data[i * n + j] = 0;
    }
  }

  MAT_FREE(V);
  MAT_FREE(T);
  MAT_FREE(W);
  MAT_FREE(tau_arr);
}

// Blocked partial pivoting LU (P * A = L * U)
// Uses cache-friendly blocking; compiler auto-vectorizes inner loops
#define MAT_PLU_BLOCK_SIZE 64

MAT_INTERNAL_STATIC int mat_plu_blocked_impl(Mat *M, Perm *p) {
  size_t n = M->rows;
  mat_elem_t *data = M->data;
  size_t *row_perm = p->data;
  int swap_count = 0;

  for (size_t kb = 0; kb < n; kb += MAT_PLU_BLOCK_SIZE) {
    size_t k_end = (kb + MAT_PLU_BLOCK_SIZE < n) ? kb + MAT_PLU_BLOCK_SIZE : n;

    // Panel factorization with pivoting
    for (size_t k = kb; k < k_end && k < n - 1; k++) {
      // Find pivot in column k
      mat_elem_t max_val = MAT_FABS(data[k * n + k]);
      size_t pivot_row = k;
      for (size_t i = k + 1; i < n; i++) {
        mat_elem_t val = MAT_FABS(data[i * n + k]);
        if (val > max_val) {
          max_val = val;
          pivot_row = i;
        }
      }

      // Swap full rows
      if (pivot_row != k) {
        mat_elem_t *row_k_ptr = &data[k * n];
        mat_elem_t *row_p_ptr = &data[pivot_row * n];
        for (size_t j = 0; j < n; j++) {
          mat_elem_t tmp = row_k_ptr[j];
          row_k_ptr[j] = row_p_ptr[j];
          row_p_ptr[j] = tmp;
        }
        size_t tmp = row_perm[k];
        row_perm[k] = row_perm[pivot_row];
        row_perm[pivot_row] = tmp;
        swap_count++;
      }

      if (MAT_FABS(data[k * n + k]) < MAT_DEFAULT_EPSILON) continue;

      mat_elem_t pivot_inv = 1.0f / data[k * n + k];
      mat_elem_t *row_k = &data[k * n];

      // Compute L column and update rows
      // Panel rows (i < k_end): full row update for correct U
      // Trailing rows (i >= k_end): panel-only update
      for (size_t i = k + 1; i < n; i++) {
        mat_elem_t *row_i = &data[i * n];
        mat_elem_t l_ik = row_i[k] * pivot_inv;
        row_i[k] = l_ik;

        size_t j_end = (i < k_end) ? n : k_end;
        for (size_t j = k + 1; j < j_end; j++)
          row_i[j] -= l_ik * row_k[j];
      }
    }

    // Trailing matrix update: A[k_end:n, k_end:n] -= L[k_end:n, kb:k_end] @ U[kb:k_end, k_end:n]
    if (k_end < n) {
      size_t trail_m = n - k_end;  // rows in trailing matrix
      size_t trail_n = n - k_end;  // cols in trailing matrix
      size_t block_k = k_end - kb; // block size
      mat_gemm_strided(
          &data[k_end * n + k_end], n,  // C: trailing submatrix
          -1.0f,                        // alpha = -1
          &data[k_end * n + kb], n,     // A: L block (rows k_end:n, cols kb:k_end)
          trail_m, block_k,             // M, K
          &data[kb * n + k_end], n,     // B: U block (rows kb:k_end, cols k_end:n)
          trail_n,                      // N
          1.0f);                        // beta = 1
    }
  }

  return swap_count;
}

// Scalar implementation of full pivoting LU (P * A * Q = L * U)
MAT_INTERNAL_STATIC int mat_lu_scalar_impl(Mat *M, Perm *p, Perm *q) {
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
MAT_INTERNAL_STATIC int mat_lu_neon_impl(Mat *M, Perm *p, Perm *q) {
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
        if (val > row_max) row_max = val;
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

  int swap_count = mat_plu_blocked_impl(M, p);

  // Extract L (lower triangular with 1s on diagonal) and U (upper triangular) from M
  mat_eye(L);
  memset(U->data, 0, n * n * sizeof(mat_elem_t));

  mat_elem_t *data = M->data;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < i; j++)
      L->data[i * n + j] = data[i * n + j];
    for (size_t j = i; j < n; j++)
      U->data[i * n + j] = data[i * n + j];
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

  // Will contain both L and U at the end)
  Mat *M = mat_rdeep_copy(A);

  // Initialize permutations to identity
  mat_perm_identity(p);
  mat_perm_identity(q);

#ifdef MAT_HAS_ARM_NEON
  int swap_count = mat_lu_neon_impl(M, p, q);
#else
  int swap_count = mat_lu_scalar_impl(M, p, q);
#endif

  // Extract L (lower triangular with 1s on diagonal) and U (upper triangular) from M
  mat_eye(L);
  memset(U->data, 0, n * n * sizeof(mat_elem_t));

  mat_elem_t *data = M->data;
  for (size_t i = 0; i < n; i++) {
    // L: copy below diagonal
    for (size_t j = 0; j < i; j++) {
      L->data[i * n + j] = data[i * n + j];
    }
    // U: copy diagonal and above
    for (size_t j = i; j < n; j++) {
      U->data[i * n + j] = data[i * n + j];
    }
  }

  MAT_FREE_MAT(M);
  return swap_count;
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

  // Y = L^-1 * P * I (forward substitution on permuted identity)
  // out stores Y, then X = U^-1 * Y
  mat_elem_t *Y = out->data;
  mat_elem_t *Ldata = L->data;
  mat_elem_t *Udata = U->data;

  // Initialize Y = P * I (permuted identity)
  memset(Y, 0, n * n * sizeof(mat_elem_t));
  for (size_t i = 0; i < n; i++)
    Y[i * n + p->data[i]] = 1;

  // Blocked forward substitution: L * Z = Y => Z = L^-1 * Y
  // Use strided GEMM for block updates
#define MAT_INV_BLOCK_SIZE 64
  for (size_t ib = 0; ib < n; ib += MAT_INV_BLOCK_SIZE) {
    size_t ie = (ib + MAT_INV_BLOCK_SIZE < n) ? ib + MAT_INV_BLOCK_SIZE : n;
    size_t block_m = ie - ib;

    // GEMM update: Y[ib:ie, :] -= L[ib:ie, 0:ib] @ Y[0:ib, :]
    if (ib > 0) {
      mat_gemm_strided(
          &Y[ib * n], n,           // C: Y[ib:ie, :], stride n
          -1.0f,                   // alpha = -1
          &Ldata[ib * n], n,       // A: L[ib:ie, 0:ib], stride n
          block_m, ib,             // M = block_m, K = ib
          Y, n, n,                 // B: Y[0:ib, :], stride n, N = n
          1.0f);                   // beta = 1
    }

    // Small triangular solve within block
    for (size_t i = ib + 1; i < ie; i++) {
      mat_elem_t *Yi = &Y[i * n];
      for (size_t j = ib; j < i; j++) {
        mat_elem_t l_ij = Ldata[i * n + j];
        mat_elem_t *Yj = &Y[j * n];
        for (size_t k = 0; k < n; k++)
          Yi[k] -= l_ij * Yj[k];
      }
    }
  }

  // Blocked backward substitution: U * X = Y => X = U^-1 * Y
  // Use strided GEMM for block updates
  for (size_t ib = n; ib > 0;) {
    size_t is = (ib > MAT_INV_BLOCK_SIZE) ? ib - MAT_INV_BLOCK_SIZE : 0;
    size_t ie = ib;
    ib = is;
    size_t block_m = ie - is;

    // GEMM update: Y[is:ie, :] -= U[is:ie, ie:n] @ Y[ie:n, :]
    if (ie < n) {
      mat_gemm_strided(
          &Y[is * n], n,           // C: Y[is:ie, :], stride n
          -1.0f,                   // alpha = -1
          &Udata[is * n + ie], n,  // A: U[is:ie, ie:n], stride n
          block_m, n - ie,         // M = block_m, K = n - ie
          &Y[ie * n], n, n,        // B: Y[ie:n, :], stride n, N = n
          1.0f);                   // beta = 1
    }

    // Small triangular solve within block (backward)
    for (size_t i = ie; i-- > is;) {
      mat_elem_t *Yi = &Y[i * n];
      mat_elem_t u_ii_inv = 1.0f / Udata[i * n + i];
      for (size_t j = i + 1; j < ie; j++) {
        mat_elem_t u_ij = Udata[i * n + j];
        mat_elem_t *Yj = &Y[j * n];
        for (size_t k = 0; k < n; k++)
          Yi[k] -= u_ij * Yj[k];
      }
      for (size_t k = 0; k < n; k++)
        Yi[k] *= u_ii_inv;
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

#endif // MAT_IMPLEMENTATION
