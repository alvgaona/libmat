#include <stdio.h>
#include <float.h>
#include <mach/mach_time.h>

#define MAT_EXPOSE_INTERNALS
#define MAT_IMPLEMENTATION
#include "mat.h"

#define ITERATIONS 100
#define ROUNDS 3

static double ns_per_tick;

void init_timer(void) {
  mach_timebase_info_data_t info;
  mach_timebase_info(&info);
  ns_per_tick = (double)info.numer / info.denom;
}

void bench_size(int size) {
  Mat *a = mat_ones(size, size);
  Mat *b = mat_ones(size, size);
  // Make them slightly different but within tolerance
  for (int i = 0; i < size * size; i += 7) {
    b->data[i] += 1e-8f;
  }

  volatile bool sink;
  for (int i = 0; i < 10; i++) {
    sink = mat_equals_tol(a, b, 1e-6f);
    sink = mat_equals_tol_neon_impl(a, b, 1e-6f);
  }

  double reg_min = DBL_MAX, neon_min = DBL_MAX;

  for (int r = 0; r < ROUNDS; r++) {
    uint64_t start = mach_absolute_time();
    for (int i = 0; i < ITERATIONS; i++) {
      sink = mat_equals_tol(a, b, 1e-6f);
    }
    uint64_t end = mach_absolute_time();
    double t = (end - start) * ns_per_tick / ITERATIONS;
    if (t < reg_min) reg_min = t;

    start = mach_absolute_time();
    for (int i = 0; i < ITERATIONS; i++) {
      sink = mat_equals_tol_neon_impl(a, b, 1e-6f);
    }
    end = mach_absolute_time();
    t = (end - start) * ns_per_tick / ITERATIONS;
    if (t < neon_min) neon_min = t;
  }

  printf("| %dx%d | %.0f ns | %.0f ns | %.1fx |\n",
         size, size, reg_min, neon_min, reg_min / neon_min);

  mat_free_mat(a);
  mat_free_mat(b);
}

int main() {
  init_timer();

  printf("## mat_equals_tol\n\n");
  printf("| Size | Regular | NEON | Speedup |\n");
  printf("|------|---------|------|--------|\n");

  int sizes[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024};
  for (int i = 0; i < 9; i++) {
    bench_size(sizes[i]);
  }

  return 0;
}
