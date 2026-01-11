#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <mach/mach_time.h>

#define MAT_IMPLEMENTATION
#include "mat.h"

#ifdef MAT_DOUBLE_PRECISION
  #define PRECISION_NAME "float64"
#else
  #define PRECISION_NAME "float32"
#endif

#define ROUNDS 20
#define WARMUP 10

static double ns_per_tick;

void init_timer(void) {
  mach_timebase_info_data_t info;
  mach_timebase_info(&info);
  ns_per_tick = (double)info.numer / info.denom;
}

typedef struct { double avg, min, std; } Stats;

Stats compute_stats(double *times, int n) {
  Stats s = {0, DBL_MAX, 0};
  for (int i = 0; i < n; i++) {
    s.avg += times[i];
    if (times[i] < s.min) s.min = times[i];
  }
  s.avg /= n;
  for (int i = 0; i < n; i++)
    s.std += (times[i] - s.avg) * (times[i] - s.avg);
  s.std = sqrt(s.std / n);
  return s;
}

void fill_random(mat_elem_t *data, size_t n) {
  for (size_t i = 0; i < n; i++)
    data[i] = (mat_elem_t)rand() / RAND_MAX * 2.0 - 1.0;
}

void bench_speed(size_t m, size_t n, int iterations) {
  Mat *A = mat_mat(m, n);
  Mat *At = mat_mat(n, m);
  fill_random(A->data, m * n);

  for (int i = 0; i < WARMUP; i++)
    mat_t(At, A);

  double times[ROUNDS];

  for (int r = 0; r < ROUNDS; r++) {
    uint64_t start = mach_absolute_time();
    for (int i = 0; i < iterations; i++)
      mat_t(At, A);
    uint64_t end = mach_absolute_time();
    times[r] = (end - start) * ns_per_tick / iterations / 1000.0;
  }

  Stats s = compute_stats(times, ROUNDS);

  // Bandwidth: read n + write n = 2n elements
  double gb = (2.0 * m * n * sizeof(mat_elem_t)) / (s.avg * 1000.0);

  printf("%4zu x %4zu: %8.2f ± %.2f us  %.1f GB/s\n", m, n, s.avg, s.std, gb);

  mat_free_mat(A); mat_free_mat(At);
}

int main() {
  srand(42); init_timer();
  printf("=== TRANSPOSE BENCHMARK: libmat [%s] ===\n", PRECISION_NAME);
  printf("B = A^T\n\n");

  bench_speed(64, 64, 10000);
  bench_speed(128, 128, 5000);
  bench_speed(256, 256, 1000);
  bench_speed(512, 512, 500);
  bench_speed(1024, 1024, 100);
  bench_speed(2048, 2048, 20);

  printf("\nNon-square:\n");
  bench_speed(1024, 64, 1000);
  bench_speed(64, 1024, 1000);

  return 0;
}
