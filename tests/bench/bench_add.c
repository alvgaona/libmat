#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include <mach/mach_time.h>

#define MAT_IMPLEMENTATION
#include "mat.h"

#ifdef MAT_DOUBLE_PRECISION
  #define PRECISION_NAME "float64"
#else
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

// Prevent compiler from optimizing away result
static void do_not_optimize(void *p) {
  __asm__ volatile("" : : "r"(p) : "memory");
}

// Simple scalar loop for comparison
void add_scalar(mat_elem_t *out, const mat_elem_t *a, const mat_elem_t *b, size_t n) {
  for (size_t i = 0; i < n; i++)
    out[i] = a[i] + b[i];
  do_not_optimize(out);
}

void bench_speed(size_t n) {
  printf("\n--- Size: %zu elements ---\n", n);

  // Use 1D layout (vector-like)
  Mat *a = mat_mat(1, n);
  Mat *b = mat_mat(1, n);
  Mat *c_libmat = mat_mat(1, n);
  mat_elem_t *c_scalar = malloc(n * sizeof(mat_elem_t));

  fill_random(a->data, n);
  fill_random(b->data, n);

  for (int i = 0; i < WARMUP; i++) {
    mat_add(c_libmat, a, b);
    add_scalar(c_scalar, a->data, b->data, n);
  }

  double libmat_times[ROUNDS], scalar_times[ROUNDS];

  for (int r = 0; r < ROUNDS; r++) {
    uint64_t start = mach_absolute_time();
    for (int i = 0; i < ITERATIONS; i++)
      mat_add(c_libmat, a, b);
    uint64_t end = mach_absolute_time();
    libmat_times[r] = (end - start) * ns_per_tick / ITERATIONS / 1000.0;

    start = mach_absolute_time();
    for (int i = 0; i < ITERATIONS; i++)
      add_scalar(c_scalar, a->data, b->data, n);
    end = mach_absolute_time();
    scalar_times[r] = (end - start) * ns_per_tick / ITERATIONS / 1000.0;
  }

  Stats ls = compute_stats(libmat_times, ROUNDS);
  Stats ss = compute_stats(scalar_times, ROUNDS);

  // Bandwidth: read 2n + write n = 3n elements
  double gb_libmat = (3.0 * n * sizeof(mat_elem_t)) / (ls.avg * 1000.0);
  double gb_scalar = (3.0 * n * sizeof(mat_elem_t)) / (ss.avg * 1000.0);

  printf("libmat:   %8.2f ± %.2f us  (%.1fx vs scalar)  %.1f GB/s\n", ls.avg, ls.std, ss.avg / ls.avg, gb_libmat);
  printf("scalar:   %8.2f ± %.2f us                     %.1f GB/s\n", ss.avg, ss.std, gb_scalar);

  mat_free_mat(a); mat_free_mat(b); mat_free_mat(c_libmat);
  free(c_scalar);
}

int main() {
  srand(42); init_timer();
  printf("=== ADD BENCHMARK: libmat vs scalar loop [%s] ===\n", PRECISION_NAME);
  printf("C = A + B (element-wise)\n");
  bench_speed(1000);
  bench_speed(10000);
  bench_speed(100000);
  bench_speed(1000000);
  bench_speed(10000000);
  return 0;
}
