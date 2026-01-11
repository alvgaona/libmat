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

  printf("Frobenius Norm Timing Comparison\n");
  printf("Matrix size: %dx%d\n", MATRIX_SIZE, MATRIX_SIZE);
  printf("Iterations per round: %d\n", ITERATIONS);
  printf("Rounds: %d\n\n", ROUNDS);

  Mat *m = mat_ones(MATRIX_SIZE, MATRIX_SIZE);

  // Warmup
  volatile float sink;
  for (int i = 0; i < WARMUP; i++) {
    sink = mat_norm_fro(m);
    sink = mat_norm_fro_neon_impl(m);
  }

  double regular_times[ROUNDS];
  double neon_times[ROUNDS];

  for (int r = 0; r < ROUNDS; r++) {
    // Time regular norm_fro
    uint64_t start = mach_absolute_time();
    for (int i = 0; i < ITERATIONS; i++) {
      sink = mat_norm_fro(m);
    }
    uint64_t end = mach_absolute_time();
    regular_times[r] = (end - start) * ns_per_tick / ITERATIONS / 1000.0;

    // Time NEON norm_fro
    start = mach_absolute_time();
    for (int i = 0; i < ITERATIONS; i++) {
      sink = mat_norm_fro_neon_impl(m);
    }
    end = mach_absolute_time();
    neon_times[r] = (end - start) * ns_per_tick / ITERATIONS / 1000.0;
  }

  // Calculate stats
  double reg_sum = 0, neon_sum = 0;
  double reg_min = DBL_MAX, neon_min = DBL_MAX;
  for (int r = 0; r < ROUNDS; r++) {
    reg_sum += regular_times[r];
    neon_sum += neon_times[r];
    if (regular_times[r] < reg_min) reg_min = regular_times[r];
    if (neon_times[r] < neon_min) neon_min = neon_times[r];
  }
  double reg_avg = reg_sum / ROUNDS;
  double neon_avg = neon_sum / ROUNDS;

  double reg_var = 0, neon_var = 0;
  for (int r = 0; r < ROUNDS; r++) {
    reg_var += (regular_times[r] - reg_avg) * (regular_times[r] - reg_avg);
    neon_var += (neon_times[r] - neon_avg) * (neon_times[r] - neon_avg);
  }
  double reg_std = sqrt(reg_var / ROUNDS);
  double neon_std = sqrt(neon_var / ROUNDS);

  printf("Regular mat_norm_fro: %.2f ± %.2f us/op (min: %.2f)\n", reg_avg, reg_std, reg_min);
  printf("NEON mat_norm_fro:    %.2f ± %.2f us/op (min: %.2f)\n", neon_avg, neon_std, neon_min);
  printf("Speedup (avg):        %.2fx\n", reg_avg / neon_avg);
  printf("Speedup (min):        %.2fx\n", reg_min / neon_min);

  // Verify correctness
  float result_regular = mat_norm_fro(m);
  float result_neon = mat_norm_fro_neon_impl(m);
  float expected = sqrtf(MATRIX_SIZE * MATRIX_SIZE); // sqrt(n) for ones matrix
  printf("\nCorrectness check:\n");
  printf("Regular: %.2f, NEON: %.2f, Expected: %.2f\n",
         result_regular, result_neon, expected);

  mat_free_mat(m);

  return 0;
}
