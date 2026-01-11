// Prototype: Cache-blocked GEMM
// Compare against current implementation

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mach/mach_time.h>

#define MAT_IMPLEMENTATION
#include "mat.h"

// Block sizes tuned for typical L2 cache (~256KB-1MB)
// Each block of A is MC x KC, B is KC x NC
// Packed A block: MC * KC * sizeof(float) should fit in L2
// Packed B block: KC * NC * sizeof(float) should fit in L2
#define MC 256   // Block rows of A (increased)
#define KC 512   // Block cols of A / rows of B (increased)
#define NC 256   // Block cols of B (increased)

// Micro-kernel tile size (must divide MC and NC)
#define MR 4
#define NR 4

// K-blocking size for simple blocking (no packing)
#define KB 256

static double ns_per_tick;

void init_timer(void) {
  mach_timebase_info_data_t info;
  mach_timebase_info(&info);
  ns_per_tick = (double)info.numer / info.denom;
}

// Pack a panel of A (mc x kc) into contiguous memory for the micro-kernel
// Packed in MR x kc strips for sequential access
static void pack_A(mat_elem_t *packed, const mat_elem_t *A,
                   size_t lda, size_t mc, size_t kc) {
  for (size_t i = 0; i < mc; i += MR) {
    size_t mr = (i + MR <= mc) ? MR : mc - i;
    for (size_t k = 0; k < kc; k++) {
      for (size_t ii = 0; ii < mr; ii++) {
        *packed++ = A[(i + ii) * lda + k];
      }
      // Pad to MR if needed
      for (size_t ii = mr; ii < MR; ii++) {
        *packed++ = 0;
      }
    }
  }
}

// Pack a panel of B (kc x nc) into contiguous memory
// Packed in kc x NR strips for sequential access
static void pack_B(mat_elem_t *packed, const mat_elem_t *B,
                   size_t ldb, size_t kc, size_t nc) {
  for (size_t j = 0; j < nc; j += NR) {
    size_t nr = (j + NR <= nc) ? NR : nc - j;
    for (size_t k = 0; k < kc; k++) {
      for (size_t jj = 0; jj < nr; jj++) {
        *packed++ = B[k * ldb + j + jj];
      }
      // Pad to NR if needed
      for (size_t jj = nr; jj < NR; jj++) {
        *packed++ = 0;
      }
    }
  }
}

#ifdef MAT_HAS_ARM_NEON
// 4x4 micro-kernel operating on packed data
// Properly vectorized: 4 row accumulators, each holds 4 column results
static void micro_kernel_4x4(size_t kc,
                             const mat_elem_t *packed_A,
                             const mat_elem_t *packed_B,
                             mat_elem_t *C, size_t ldc,
                             mat_elem_t alpha) {
  // 4 accumulators, one per row of C tile (each holds 4 columns)
  float32x4_t c0 = vdupq_n_f32(0);
  float32x4_t c1 = vdupq_n_f32(0);
  float32x4_t c2 = vdupq_n_f32(0);
  float32x4_t c3 = vdupq_n_f32(0);

  // packed_A: MR values per k (a0, a1, a2, a3 for each k)
  // packed_B: NR values per k (b0, b1, b2, b3 for each k)
  for (size_t k = 0; k < kc; k++) {
    // Load 4 A values and 4 B values
    float32x4_t a = vld1q_f32(&packed_A[k * MR]);
    float32x4_t b = vld1q_f32(&packed_B[k * NR]);

    // Rank-1 update: C += a * b^T
    // Each row of C gets a[i] * b (broadcast a[i] and multiply by b vector)
    c0 = vfmaq_laneq_f32(c0, b, a, 0);  // c0 += a[0] * b
    c1 = vfmaq_laneq_f32(c1, b, a, 1);  // c1 += a[1] * b
    c2 = vfmaq_laneq_f32(c2, b, a, 2);  // c2 += a[2] * b
    c3 = vfmaq_laneq_f32(c3, b, a, 3);  // c3 += a[3] * b
  }

  // Scale by alpha
  float32x4_t valpha = vdupq_n_f32(alpha);
  c0 = vmulq_f32(c0, valpha);
  c1 = vmulq_f32(c1, valpha);
  c2 = vmulq_f32(c2, valpha);
  c3 = vmulq_f32(c3, valpha);

  // Load existing C values, add our results, store back
  float32x4_t r0 = vld1q_f32(&C[0*ldc]);
  float32x4_t r1 = vld1q_f32(&C[1*ldc]);
  float32x4_t r2 = vld1q_f32(&C[2*ldc]);
  float32x4_t r3 = vld1q_f32(&C[3*ldc]);

  vst1q_f32(&C[0*ldc], vaddq_f32(r0, c0));
  vst1q_f32(&C[1*ldc], vaddq_f32(r1, c1));
  vst1q_f32(&C[2*ldc], vaddq_f32(r2, c2));
  vst1q_f32(&C[3*ldc], vaddq_f32(r3, c3));
}
#else
static void micro_kernel_4x4(size_t kc,
                             const mat_elem_t *packed_A,
                             const mat_elem_t *packed_B,
                             mat_elem_t *C, size_t ldc,
                             mat_elem_t alpha) {
  mat_elem_t acc[MR][NR] = {0};

  for (size_t k = 0; k < kc; k++) {
    for (size_t i = 0; i < MR; i++) {
      mat_elem_t a = packed_A[k * MR + i];
      for (size_t j = 0; j < NR; j++) {
        acc[i][j] += a * packed_B[k * NR + j];
      }
    }
  }

  for (size_t i = 0; i < MR; i++) {
    for (size_t j = 0; j < NR; j++) {
      C[i * ldc + j] += alpha * acc[i][j];
    }
  }
}
#endif

// Cache-blocked GEMM: C = alpha * A * B + beta * C
void gemm_blocked(size_t M, size_t N, size_t K,
                  mat_elem_t alpha,
                  const mat_elem_t *A, size_t lda,
                  const mat_elem_t *B, size_t ldb,
                  mat_elem_t beta,
                  mat_elem_t *C, size_t ldc) {
  // Scale C by beta
  if (beta == 0) {
    for (size_t i = 0; i < M; i++)
      memset(&C[i * ldc], 0, N * sizeof(mat_elem_t));
  } else if (beta != 1) {
    for (size_t i = 0; i < M; i++)
      for (size_t j = 0; j < N; j++)
        C[i * ldc + j] *= beta;
  }

  // Static packing buffers (avoid malloc overhead in hot path)
  static mat_elem_t packed_A[MC * KC];
  static mat_elem_t packed_B[KC * NC];

  // Loop over K in blocks of KC
  for (size_t kk = 0; kk < K; kk += KC) {
    size_t kc = (kk + KC <= K) ? KC : K - kk;

    // Loop over N in blocks of NC
    for (size_t jj = 0; jj < N; jj += NC) {
      size_t nc = (jj + NC <= N) ? NC : N - jj;

      // Pack B panel (kc x nc)
      pack_B(packed_B, &B[kk * ldb + jj], ldb, kc, nc);

      // Loop over M in blocks of MC
      for (size_t ii = 0; ii < M; ii += MC) {
        size_t mc = (ii + MC <= M) ? MC : M - ii;

        // Pack A panel (mc x kc)
        pack_A(packed_A, &A[ii * lda + kk], lda, mc, kc);

        // Compute C[ii:ii+mc, jj:jj+nc] += A_packed * B_packed
        for (size_t i = 0; i < mc; i += MR) {
          for (size_t j = 0; j < nc; j += NR) {
            micro_kernel_4x4(kc,
                            &packed_A[i * kc],
                            &packed_B[j * kc],
                            &C[(ii + i) * ldc + jj + j], ldc,
                            alpha);
          }
        }
      }
    }
  }

}

void fill_random(mat_elem_t *data, size_t n) {
  for (size_t i = 0; i < n; i++)
    data[i] = (mat_elem_t)rand() / RAND_MAX * 2.0 - 1.0;
}

double bench(size_t n, int iterations) {
  Mat *A = mat_mat(n, n);
  Mat *B = mat_mat(n, n);
  Mat *C1 = mat_mat(n, n);
  Mat *C2 = mat_mat(n, n);

  fill_random(A->data, n * n);
  fill_random(B->data, n * n);

  // Warmup
  mat_gemm(C1, 1.0, A, B, 0.0);
  gemm_blocked(n, n, n, 1.0, A->data, n, B->data, n, 0.0, C2->data, n);

  // Verify correctness
  mat_elem_t max_diff = 0;
  for (size_t i = 0; i < n * n; i++) {
    mat_elem_t diff = fabsf(C1->data[i] - C2->data[i]);
    if (diff > max_diff) max_diff = diff;
  }

  // Benchmark current
  memset(C1->data, 0, n * n * sizeof(mat_elem_t));
  uint64_t start = mach_absolute_time();
  for (int i = 0; i < iterations; i++)
    mat_gemm(C1, 1.0, A, B, 0.0);
  uint64_t end = mach_absolute_time();
  double current_us = (end - start) * ns_per_tick / iterations / 1000.0;

  // Benchmark blocked
  memset(C2->data, 0, n * n * sizeof(mat_elem_t));
  start = mach_absolute_time();
  for (int i = 0; i < iterations; i++)
    gemm_blocked(n, n, n, 1.0, A->data, n, B->data, n, 0.0, C2->data, n);
  end = mach_absolute_time();
  double blocked_us = (end - start) * ns_per_tick / iterations / 1000.0;

  double gflops_current = (2.0 * n * n * n) / (current_us * 1000.0);
  double gflops_blocked = (2.0 * n * n * n) / (blocked_us * 1000.0);

  printf("%4zu x %4zu: current %8.1f us (%5.1f GFLOPS) | blocked %8.1f us (%5.1f GFLOPS) | %.2fx | diff: %e\n",
         n, n, current_us, gflops_current, blocked_us, gflops_blocked,
         current_us / blocked_us, max_diff);

  mat_free_mat(A);
  mat_free_mat(B);
  mat_free_mat(C1);
  mat_free_mat(C2);

  return current_us / blocked_us;
}

int main() {
  srand(42);
  init_timer();

  printf("=== GEMM Blocked Prototype ===\n");
  printf("Block sizes: MC=%d, KC=%d, NC=%d, MR=%d, NR=%d\n\n", MC, KC, NC, MR, NR);

  bench(64, 1000);
  bench(128, 500);
  bench(256, 100);
  bench(512, 20);
  bench(1024, 5);
  bench(2048, 2);

  return 0;
}
