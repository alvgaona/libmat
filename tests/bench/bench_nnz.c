#include <stdio.h>
#include <float.h>
#include <mach/mach_time.h>

#define MAT_EXPOSE_INTERNALS
#define MAT_IMPLEMENTATION
#include "../../mat.h"

#define ITERATIONS 100
#define ROUNDS 3

static double ns_per_tick;

void init_timer(void) {
  mach_timebase_info_data_t info;
  mach_timebase_info(&info);
  ns_per_tick = (double)info.numer / info.denom;
}

void bench_size(int size) {
  Mat *m = mat_ones(size, size);
  for (int i = 0; i < size * size / 2; i += 3) {
    m->data[i] = 0;
  }

  volatile float sink;
  for (int i = 0; i < 10; i++) {
    sink = mat_nnz(m);
    sink = mat_nnz_neon_impl(m);
  }

  double reg_min = DBL_MAX, neon_min = DBL_MAX;

  for (int r = 0; r < ROUNDS; r++) {
    uint64_t start = mach_absolute_time();
    for (int i = 0; i < ITERATIONS; i++) {
      sink = mat_nnz(m);
    }
    uint64_t end = mach_absolute_time();
    double t = (end - start) * ns_per_tick / ITERATIONS;
    if (t < reg_min) reg_min = t;

    start = mach_absolute_time();
    for (int i = 0; i < ITERATIONS; i++) {
      sink = mat_nnz_neon_impl(m);
    }
    end = mach_absolute_time();
    t = (end - start) * ns_per_tick / ITERATIONS;
    if (t < neon_min) neon_min = t;
  }

  printf("%4dx%-4d  %8.1f ns  %8.1f ns  %5.2fx\n",
         size, size, reg_min, neon_min, reg_min / neon_min);

  mat_free_mat(m);
}

int main() {
  init_timer();

  printf("Size      Regular      NEON     Speedup\n");
  printf("----------------------------------------\n");

  int sizes[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024};
  for (int i = 0; i < 9; i++) {
    bench_size(sizes[i]);
  }

  return 0;
}
