/*
  bench.h - stb-style single-header benchmarking utilities

  Do this:
    #define BENCH_IMPLEMENTATION
  before you include this file in *one* C/C++ file to create the implementation.

  // example usage:
  #define BENCH_IMPLEMENTATION
  #include "bench.h"

  int main() {
    bench_init();
    uint64_t start = bench_now();
    // ... code to benchmark ...
    uint64_t end = bench_now();
    printf("%.2f ns\n", bench_ns(start, end));
  }

  Configuration (define before including):
    #define BENCH_ITERATIONS 100   // iterations per round
    #define BENCH_ROUNDS 5         // number of rounds
    #define BENCH_WARMUP 3         // warmup iterations
*/

#ifndef BENCH_H
#define BENCH_H

#include <stdint.h>
#include <stddef.h>

// ============ Configuration ============

#ifndef BENCH_ITERATIONS
#define BENCH_ITERATIONS 100
#endif

#ifndef BENCH_ROUNDS
#define BENCH_ROUNDS 5
#endif

#ifndef BENCH_WARMUP
#define BENCH_WARMUP 3
#endif

// ============ API ============

typedef struct {
  double avg;
  double min;
  double std;
} BenchStats;

#ifdef __cplusplus
extern "C" {
#endif

void        bench_init(void);
uint64_t    bench_now(void);
double      bench_ns(uint64_t start, uint64_t end);
BenchStats  bench_stats(const double *times, int n);
void        bench_fill_random_f(float *data, size_t n);
void        bench_fill_random_d(double *data, size_t n);

// Run benchmark: returns best time in ns
typedef void (*bench_void_fn)(void);
double      bench_run(bench_void_fn fn, int iters);

#ifdef __cplusplus
}
#endif

#endif // BENCH_H

// ============ Implementation ============

#ifdef BENCH_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
static double bench_ns_per_tick_;
void bench_init(void) {
  mach_timebase_info_data_t info;
  mach_timebase_info(&info);
  bench_ns_per_tick_ = (double)info.numer / info.denom;
}
uint64_t bench_now(void) { return mach_absolute_time(); }
double bench_ns(uint64_t start, uint64_t end) { return (end - start) * bench_ns_per_tick_; }
#else
#include <time.h>
void bench_init(void) {}
uint64_t bench_now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}
double bench_ns(uint64_t start, uint64_t end) { return (double)(end - start); }
#endif

BenchStats bench_stats(const double *times, int n) {
  BenchStats s = {0, DBL_MAX, 0};
  for (int i = 0; i < n; i++) {
    s.avg += times[i];
    if (times[i] < s.min) s.min = times[i];
  }
  s.avg /= n;
  for (int i = 0; i < n; i++) {
    double d = times[i] - s.avg;
    s.std += d * d;
  }
  s.std = sqrt(s.std / n);
  return s;
}

void bench_fill_random_f(float *data, size_t n) {
  for (size_t i = 0; i < n; i++)
    data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

void bench_fill_random_d(double *data, size_t n) {
  for (size_t i = 0; i < n; i++)
    data[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
}

double bench_run(bench_void_fn fn, int iters) {
  for (int i = 0; i < BENCH_WARMUP; i++) fn();
  double best = DBL_MAX;
  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < iters; i++) fn();
    uint64_t end = bench_now();
    double t = bench_ns(start, end) / iters;
    if (t < best) best = t;
  }
  return best;
}

#endif // BENCH_IMPLEMENTATION
