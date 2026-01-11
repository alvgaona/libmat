#include <stdio.h>
#include <float.h>
#include <math.h>
#include <mach/mach_time.h>

#define MAT_EXPOSE_INTERNALS
#define MAT_IMPLEMENTATION
#include "mat.h"

#define MATRIX_SIZE 1024
#define ITERATIONS 1000
#define ROUNDS 20
#define WARMUP 10

static double ns_per_tick;

void init_timer(void) {
  mach_timebase_info_data_t info;
  mach_timebase_info(&info);
  ns_per_tick = (double)info.numer / info.denom;
}

int main() {
  init_timer();

  printf("Max Norm Timing Comparison\n");
  printf("Matrix size: %dx%d\n", MATRIX_SIZE, MATRIX_SIZE);
  printf("Rounds: %d\n\n", ROUNDS);

  Mat *m = mat_ones(MATRIX_SIZE, MATRIX_SIZE);
  // Set one element to a larger value for correctness check
  m->data[500] = 42.0f;

  // Warmup
  volatile float sink;
  for (int i = 0; i < WARMUP; i++) {
    sink = mat_norm_max(m);
    sink = mat_norm_max_neon_impl(m);
  }

  double regular_times[ROUNDS];
  double neon_times[ROUNDS];

  for (int r = 0; r < ROUNDS; r++) {
    uint64_t start = mach_absolute_time();
    for (int i = 0; i < ITERATIONS; i++) {
      sink = mat_norm_max(m);
    }
    uint64_t end = mach_absolute_time();
    regular_times[r] = (end - start) * ns_per_tick / ITERATIONS / 1000.0;

    start = mach_absolute_time();
    for (int i = 0; i < ITERATIONS; i++) {
      sink = mat_norm_max_neon_impl(m);
    }
    end = mach_absolute_time();
    neon_times[r] = (end - start) * ns_per_tick / ITERATIONS / 1000.0;
  }

  double reg_min = DBL_MAX, neon_min = DBL_MAX;
  for (int r = 0; r < ROUNDS; r++) {
    if (regular_times[r] < reg_min) reg_min = regular_times[r];
    if (neon_times[r] < neon_min) neon_min = neon_times[r];
  }

  printf("Regular mat_norm_max: %.2f us/op (min)\n", reg_min);
  printf("NEON mat_norm_max:    %.2f us/op (min)\n", neon_min);
  printf("Speedup (min):        %.2fx\n", reg_min / neon_min);

  // Verify correctness
  float result_regular = mat_norm_max(m);
  float result_neon = mat_norm_max_neon_impl(m);
  printf("\nCorrectness: Regular=%.0f, NEON=%.0f, Expected=42\n",
         result_regular, result_neon);

  mat_free_mat(m);

  return 0;
}
