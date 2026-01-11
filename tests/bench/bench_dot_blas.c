#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <mach/mach_time.h>
#include <cblas.h>

#define MAT_IMPLEMENTATION
#include "../../mat.h"

#ifdef MAT_DOUBLE_PRECISION
  #define BLAS_DOT cblas_ddot
  #define PRECISION_NAME "float64"
#else
  #define BLAS_DOT cblas_sdot
  #define PRECISION_NAME "float32"
#endif

#define ITERATIONS 1000
#define ROUNDS 20
#define WARMUP 10

static double ns_per_tick;

void init_timer(void) {
  mach_timebase_info_data_t info;
  mach_timebase_info(&info);
  ns_per_tick = (double)info.numer / info.denom;
}

typedef struct {
  double avg, min, std;
} Stats;

Stats compute_stats(double *times, int n) {
  Stats s;
  double sum = 0;
  s.min = DBL_MAX;
  for (int i = 0; i < n; i++) {
    sum += times[i];
    if (times[i] < s.min) s.min = times[i];
  }
  s.avg = sum / n;
  double var = 0;
  for (int i = 0; i < n; i++) {
    var += (times[i] - s.avg) * (times[i] - s.avg);
  }
  s.std = sqrt(var / n);
  return s;
}

void fill_random(Mat *v) {
  for (size_t i = 0; i < v->rows * v->cols; i++) {
    v->data[i] = (mat_elem_t)rand() / RAND_MAX * 2.0 - 1.0;
  }
}

void bench_speed(size_t size) {
  printf("\n--- Vector size: %zu ---\n", size);

  Vec *a = mat_vec(size);
  Vec *b = mat_vec(size);
  fill_random(a);
  fill_random(b);

  volatile mat_elem_t sink;
  for (int i = 0; i < WARMUP; i++) {
    sink = mat_dot(a, b);
    sink = BLAS_DOT((int)size, a->data, 1, b->data, 1);
  }

  double libmat_times[ROUNDS];
  double blas_times[ROUNDS];

  for (int r = 0; r < ROUNDS; r++) {
    uint64_t start = mach_absolute_time();
    for (int i = 0; i < ITERATIONS; i++) {
      sink = mat_dot(a, b);
    }
    uint64_t end = mach_absolute_time();
    libmat_times[r] = (end - start) * ns_per_tick / ITERATIONS / 1000.0;

    start = mach_absolute_time();
    for (int i = 0; i < ITERATIONS; i++) {
      sink = BLAS_DOT((int)size, a->data, 1, b->data, 1);
    }
    end = mach_absolute_time();
    blas_times[r] = (end - start) * ns_per_tick / ITERATIONS / 1000.0;
  }

  Stats libmat_s = compute_stats(libmat_times, ROUNDS);
  Stats blas_s = compute_stats(blas_times, ROUNDS);

  printf("libmat:   %8.2f ± %.2f us  (%.1fx vs BLAS)\n",
         libmat_s.avg, libmat_s.std, blas_s.avg / libmat_s.avg);
  printf("OpenBLAS: %8.2f ± %.2f us\n",
         blas_s.avg, blas_s.std);

  mat_free_mat(a);
  mat_free_mat(b);
}

int main() {
  srand(42);
  init_timer();

  printf("=== DOT PRODUCT BENCHMARK: libmat vs OpenBLAS [%s] ===\n", PRECISION_NAME);
  printf("Iterations per round: %d\n", ITERATIONS);
  printf("Rounds: %d\n", ROUNDS);

  bench_speed(100);
  bench_speed(1000);
  bench_speed(10000);
  bench_speed(100000);
  bench_speed(1000000);

  return 0;
}
