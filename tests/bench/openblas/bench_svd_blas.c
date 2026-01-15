#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>
#include <lapacke.h>

// Force single-threaded before OpenBLAS initializes
__attribute__((constructor))
static void force_single_thread(void) {
  setenv("OPENBLAS_NUM_THREADS", "1", 1);
  setenv("OMP_NUM_THREADS", "1", 1);
  setenv("GOTO_NUM_THREADS", "1", 1);
}

#define BENCH_ROUNDS 10
#define BENCH_WARMUP 3
#define BENCH_IMPLEMENTATION
#include "bench.h"

#define MAT_IMPLEMENTATION
#include "mat.h"

#ifdef MAT_DOUBLE_PRECISION
  #define LAPACK_GESVD LAPACKE_dgesvd
  #define LAPACK_GESDD LAPACKE_dgesdd
  #define LAPACK_GESVJ LAPACKE_dgesvj
  #define PRECISION_NAME "float64"
  #define BENCH_FILL bench_fill_random_d
#else
  #define LAPACK_GESVD LAPACKE_sgesvd
  #define LAPACK_GESDD LAPACKE_sgesdd
  #define LAPACK_GESVJ LAPACKE_sgesvj
  #define PRECISION_NAME "float32"
  #define BENCH_FILL bench_fill_random_f
#endif

void bench_svd(size_t m, size_t n, int iterations) {
  size_t k = m < n ? m : n;
  printf("\n--- %zux%zu ---\n", m, n);

  // Allocate for libmat
  Mat *A = mat_mat(m, n);
  Mat *U = mat_mat(m, m);
  Vec *S = mat_vec(k);
  Mat *Vt = mat_mat(n, n);

  // Allocate for LAPACK
  mat_elem_t *A_lap = malloc(m * n * sizeof(mat_elem_t));
  mat_elem_t *U_lap = malloc(m * m * sizeof(mat_elem_t));
  mat_elem_t *S_lap = malloc(k * sizeof(mat_elem_t));
  mat_elem_t *Vt_lap = malloc(n * n * sizeof(mat_elem_t));
  mat_elem_t *superb = malloc(k * sizeof(mat_elem_t));
  mat_elem_t *stat = malloc(6 * sizeof(mat_elem_t));  // for gesvj

  // Original data
  mat_elem_t *A_orig = malloc(m * n * sizeof(mat_elem_t));
  BENCH_FILL(A_orig, m * n);

  lapack_int mm = (lapack_int)m;
  lapack_int nn = (lapack_int)n;

  // Warmup
  for (int i = 0; i < BENCH_WARMUP; i++) {
    memcpy(A->data, A_orig, m * n * sizeof(mat_elem_t));
    mat_svd(A, U, S, Vt);

    memcpy(A_lap, A_orig, m * n * sizeof(mat_elem_t));
    LAPACK_GESVD(LAPACK_ROW_MAJOR, 'A', 'A', mm, nn, A_lap, nn, S_lap, U_lap, mm, Vt_lap, nn, superb);
  }

  double libmat_times[BENCH_ROUNDS], gesvj_times[BENCH_ROUNDS], gesdd_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    // libmat (Jacobi)
    uint64_t start = bench_now();
    for (int i = 0; i < iterations; i++) {
      memcpy(A->data, A_orig, m * n * sizeof(mat_elem_t));
      mat_svd(A, U, S, Vt);
    }
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / iterations / 1000.0;

    // LAPACK gesvj (Jacobi) - only works for m >= n
    if (m >= n) {
      start = bench_now();
      for (int i = 0; i < iterations; i++) {
        memcpy(A_lap, A_orig, m * n * sizeof(mat_elem_t));
        // joba='G' (general), jobu='U' (compute U in A), jobv='V' (compute V)
        LAPACK_GESVJ(LAPACK_ROW_MAJOR, 'G', 'U', 'V', mm, nn, A_lap, nn, S_lap, nn, Vt_lap, nn, stat);
      }
      end = bench_now();
      gesvj_times[r] = bench_ns(start, end) / iterations / 1000.0;
    }

    // LAPACK gesdd (divide & conquer)
    start = bench_now();
    for (int i = 0; i < iterations; i++) {
      memcpy(A_lap, A_orig, m * n * sizeof(mat_elem_t));
      LAPACK_GESDD(LAPACK_ROW_MAJOR, 'A', mm, nn, A_lap, nn, S_lap, U_lap, mm, Vt_lap, nn);
    }
    end = bench_now();
    gesdd_times[r] = bench_ns(start, end) / iterations / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats ds = bench_stats(gesdd_times, BENCH_ROUNDS);

  printf("libmat (Jacobi): %10.1f ± %5.1f us\n", ls.avg, ls.std);
  if (m >= n) {
    BenchStats js = bench_stats(gesvj_times, BENCH_ROUNDS);
    printf("LAPACK (gesvj):  %10.1f ± %5.1f us  (%.1fx)\n", js.avg, js.std, ls.avg / js.avg);
  }
  printf("LAPACK (gesdd):  %10.1f ± %5.1f us  (%.1fx)\n", ds.avg, ds.std, ls.avg / ds.avg);

  mat_free_mat(A);
  mat_free_mat(U);
  mat_free_mat(S);
  mat_free_mat(Vt);
  free(A_lap);
  free(U_lap);
  free(S_lap);
  free(Vt_lap);
  free(superb);
  free(stat);
  free(A_orig);
}

int main() {
  srand(42);
  bench_init();

  printf("=== SVD BENCHMARK: libmat vs LAPACK [%s] ===\n", PRECISION_NAME);
  printf("libmat: one-sided Jacobi\n");
  printf("gesvj:  LAPACK Jacobi (m >= n only)\n");
  printf("gesdd:  bidiag + divide & conquer\n");
  printf("Rounds: %d\n", BENCH_ROUNDS);
  printf("OpenBLAS threads: %d\n", openblas_get_num_threads());

  bench_svd(10, 10, 1000);
  bench_svd(20, 20, 500);
  bench_svd(50, 50, 100);
  bench_svd(100, 100, 20);
  bench_svd(200, 200, 5);
  bench_svd(50, 30, 100);   // tall
  bench_svd(30, 50, 100);   // wide

  return 0;
}
