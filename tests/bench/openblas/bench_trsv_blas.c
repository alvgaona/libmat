#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

// Force single-threaded before OpenBLAS initializes
__attribute__((constructor))
static void force_single_thread(void) {
  setenv("OPENBLAS_NUM_THREADS", "1", 1);
  setenv("OMP_NUM_THREADS", "1", 1);
  setenv("GOTO_NUM_THREADS", "1", 1);
}

#define BENCH_ROUNDS 20
#define BENCH_WARMUP 5
#define BENCH_IMPLEMENTATION
#include "bench.h"

#define MAT_IMPLEMENTATION
#include "mat.h"

#ifdef MAT_DOUBLE_PRECISION
  #define BLAS_TRSV cblas_dtrsv
  #define PRECISION_NAME "float64"
  #define BENCH_FILL bench_fill_random_d
#else
  #define BLAS_TRSV cblas_strsv
  #define PRECISION_NAME "float32"
  #define BENCH_FILL bench_fill_random_f
#endif

// Create well-conditioned lower triangular matrix
void make_tril(mat_elem_t *L, size_t n) {
  for (size_t i = 0; i < n; i++) {
    mat_elem_t row_sum = 0;
    for (size_t j = 0; j < i; j++) {
      mat_elem_t val = (mat_elem_t)(rand() % 100) / 100.0f - 0.5f;
      L[i * n + j] = val;
      row_sum += (val > 0 ? val : -val);
    }
    L[i * n + i] = row_sum + 1.0f;  // diagonally dominant
    for (size_t j = i + 1; j < n; j++) {
      L[i * n + j] = 0;
    }
  }
}

void bench_tril(size_t n, int iterations) {
  printf("\n--- Lower triangular %zux%zu ---\n", n, n);

  Mat *L = mat_mat(n, n);
  Vec *b = mat_vec(n);
  Vec *x = mat_vec(n);
  Vec *x_blas = mat_vec(n);

  make_tril(L->data, n);
  BENCH_FILL(b->data, n);

  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_solve_tril(x, L, b);
    mat_deep_copy(x_blas, b);
    BLAS_TRSV(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
              (int)n, L->data, (int)n, x_blas->data, 1);
  }

  double libmat_times[BENCH_ROUNDS], blas_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < iterations; i++)
      mat_solve_tril(x, L, b);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / iterations / 1000.0;

    start = bench_now();
    for (int i = 0; i < iterations; i++) {
      mat_deep_copy(x_blas, b);
      BLAS_TRSV(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                (int)n, L->data, (int)n, x_blas->data, 1);
    }
    end = bench_now();
    blas_times[r] = bench_ns(start, end) / iterations / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats bs = bench_stats(blas_times, BENCH_ROUNDS);

  printf("libmat:   %8.2f ± %.2f us  (%.2fx vs BLAS)\n",
         ls.avg, ls.std, bs.avg / ls.avg);
  printf("OpenBLAS: %8.2f ± %.2f us\n", bs.avg, bs.std);

  mat_free_mat(L);
  mat_free_mat(b);
  mat_free_mat(x);
  mat_free_mat(x_blas);
}

void bench_triu(size_t n, int iterations) {
  printf("\n--- Upper triangular %zux%zu ---\n", n, n);

  Mat *U = mat_mat(n, n);
  Vec *b = mat_vec(n);
  Vec *x = mat_vec(n);
  Vec *x_blas = mat_vec(n);

  // Create upper triangular (transpose of lower)
  make_tril(U->data, n);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = i + 1; j < n; j++) {
      U->data[i * n + j] = U->data[j * n + i];
      U->data[j * n + i] = 0;
    }
  }
  BENCH_FILL(b->data, n);

  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_solve_triu(x, U, b);
    mat_deep_copy(x_blas, b);
    BLAS_TRSV(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
              (int)n, U->data, (int)n, x_blas->data, 1);
  }

  double libmat_times[BENCH_ROUNDS], blas_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < iterations; i++)
      mat_solve_triu(x, U, b);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / iterations / 1000.0;

    start = bench_now();
    for (int i = 0; i < iterations; i++) {
      mat_deep_copy(x_blas, b);
      BLAS_TRSV(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                (int)n, U->data, (int)n, x_blas->data, 1);
    }
    end = bench_now();
    blas_times[r] = bench_ns(start, end) / iterations / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats bs = bench_stats(blas_times, BENCH_ROUNDS);

  printf("libmat:   %8.2f ± %.2f us  (%.2fx vs BLAS)\n",
         ls.avg, ls.std, bs.avg / ls.avg);
  printf("OpenBLAS: %8.2f ± %.2f us\n", bs.avg, bs.std);

  mat_free_mat(U);
  mat_free_mat(b);
  mat_free_mat(x);
  mat_free_mat(x_blas);
}

int main() {
  srand(42);
  bench_init();

  bench_print_summary("libmat vs OpenBLAS: TRSV");
  printf("Precision: %s\n", PRECISION_NAME);
  printf("Solve Tx = b (triangular solve)\n");
  printf("Rounds: %d, OpenBLAS threads: %d\n", BENCH_ROUNDS, openblas_get_num_threads());

  printf("\n========== Lower Triangular (Lx = b) ==========\n");
  bench_tril(32, 5000);
  bench_tril(64, 2000);
  bench_tril(128, 500);
  bench_tril(256, 200);
  bench_tril(512, 50);

  printf("\n========== Upper Triangular (Ux = b) ==========\n");
  bench_triu(32, 5000);
  bench_triu(64, 2000);
  bench_triu(128, 500);
  bench_triu(256, 200);
  bench_triu(512, 50);

  return 0;
}
