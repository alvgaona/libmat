/*
 * zap.h - STB-style single-header benchmarking library for C
 *
 * Inspired by criterion-rs (Rust benchmarking framework)
 *
 * USAGE:
 *   In exactly ONE C file, before including this header:
 *     #define ZAP_IMPLEMENTATION
 *     #include "zap.h"
 *
 *   In all other files, just:
 *     #include "zap.h"
 *
 * EXAMPLE:
 *   void bench_example(zap_t* c) {
 *       ZAP_LOOP(c) {
 *           // code to benchmark
 *       }
 *   }
 *
 *   ZAP_GROUP(my_benches, bench_example);
 *   ZAP_MAIN(my_benches);
 *
 * LICENSE: MIT (see end of file)
 */

#ifndef ZAP_H
#define ZAP_H

#ifdef __cplusplus
extern "C" {
#endif

/* CONFIGURATION */

// Measurement defaults
#ifndef ZAP_DEFAULT_SAMPLE_COUNT
#define ZAP_DEFAULT_SAMPLE_COUNT 100
#endif

#ifndef ZAP_DEFAULT_WARMUP_TIME_NS
#define ZAP_DEFAULT_WARMUP_TIME_NS 1000000000ULL  // 1 second
#endif

#ifndef ZAP_DEFAULT_MEASUREMENT_TIME_NS
#define ZAP_DEFAULT_MEASUREMENT_TIME_NS 3000000000ULL  // 3 seconds
#endif

#ifndef ZAP_DEFAULT_MIN_ITERS
#define ZAP_DEFAULT_MIN_ITERS 0  // 0 = auto
#endif

#ifndef ZAP_CONFIDENCE_LEVEL
#define ZAP_CONFIDENCE_LEVEL 0.95
#endif

// Output defaults (0 = off, 1 = on)
#ifndef ZAP_DEFAULT_SHOW_ENV
#define ZAP_DEFAULT_SHOW_ENV 0
#endif

#ifndef ZAP_DEFAULT_SHOW_HISTOGRAM
#define ZAP_DEFAULT_SHOW_HISTOGRAM 0
#endif

#ifndef ZAP_DEFAULT_SHOW_PERCENTILES
#define ZAP_DEFAULT_SHOW_PERCENTILES 0
#endif

// Color mode: 0 = auto, 1 = always, 2 = never
#ifndef ZAP_DEFAULT_COLOR_MODE
#define ZAP_DEFAULT_COLOR_MODE 0
#endif

/* INCLUDES */

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* TYPES */

// Throughput type
typedef enum zap_throughput_type {
    ZAP_THROUGHPUT_NONE = 0,
    ZAP_THROUGHPUT_BYTES,
    ZAP_THROUGHPUT_ELEMENTS
} zap_throughput_type_t;

// Statistics results
typedef struct zap_stats {
    double mean;
    double median;           // Same as p50
    double std_dev;
    double mad;              // Median Absolute Deviation
    double ci_lower;         // Confidence interval lower bound
    double ci_upper;         // Confidence interval upper bound
    double min;              // Minimum sample value
    double max;              // Maximum sample value
    double p75;              // 75th percentile
    double p90;              // 90th percentile
    double p95;              // 95th percentile
    double p99;              // 99th percentile
    size_t outliers_low;     // Number of low outliers
    size_t outliers_high;    // Number of high outliers
    size_t sample_count;
    size_t iterations;       // Iterations per sample
    double* samples;         // Pointer to samples for histogram
    // Throughput info
    zap_throughput_type_t throughput_type;
    size_t throughput_value; /* Bytes or elements per iteration */
} zap_stats_t;

// Per-benchmark configuration
typedef struct zap_bench_config {
    uint64_t warmup_time_ns;
    uint64_t measurement_time_ns;
    size_t   sample_count;
} zap_bench_config_t;

// Benchmark state
typedef struct criterion {
    const char* name;
    uint64_t    iterations;
    uint64_t    current_iter;
    uint64_t    start_time;
    double*     samples;
    size_t      sample_count;
    size_t      sample_capacity;
    bool        warmup_complete;
    bool        measuring;
    bool        status_printed;  // Track if warmup/measuring status was shown
    zap_bench_config_t config;
    // Throughput tracking
    zap_throughput_type_t throughput_type;
    size_t throughput_value;
} zap_t;

// Forward declaration for bencher
typedef struct zap_bencher zap_bencher_t;

// Benchmark function signatures
typedef void (*zap_bench_fn)(zap_t*);
typedef void (*zap_bencher_fn)(zap_bencher_t*);
typedef void (*zap_param_fn)(zap_bencher_t*, void* param);

// Benchmark entry
typedef struct zap_bench {
    const char*        name;
    zap_bench_fn func;
} zap_bench_t;

// Static benchmark group (from ZAP_GROUP macro)
typedef struct zap_group {
    const char*             name;
    const zap_bench_t* benches;
    size_t                  count;
} zap_group_t;

// Setup/teardown function types
typedef void (*zap_setup_fn)(void);
typedef void (*zap_teardown_fn)(void);

// Maximum tags per benchmark group
#ifndef ZAP_MAX_TAGS
#define ZAP_MAX_TAGS 8
#endif

// Runtime benchmark group
typedef struct zap_runtime_group {
    char                     name[128];
    zap_bench_config_t config;
    bool                     active;
    bool                     header_printed;  // Deferred header for filtering
    zap_setup_fn             setup;           // Called before group runs
    zap_teardown_fn          teardown;        // Called after group completes
    // Tags for filtering
    char                     tags[ZAP_MAX_TAGS][32];
    size_t                   tag_count;
} zap_runtime_group_t;

// Bencher - passed to iter() style benchmarks
struct zap_bencher {
    zap_t*              state;
    zap_runtime_group_t* group;
    char                      full_name[256];
};

// Benchmark ID for parameterized benchmarks
typedef struct zap_benchmark_id {
    char label[128];
    char param_str[64];
} zap_benchmark_id_t;

// Comparison result
typedef enum zap_change {
    ZAP_NO_CHANGE,
    ZAP_IMPROVED,
    ZAP_REGRESSED
} zap_change_t;

// Baseline entry for a single benchmark
typedef struct zap_baseline_entry {
    char                name[256];
    double              mean;
    double              std_dev;
    double              ci_lower;
    double              ci_upper;
} zap_baseline_entry_t;

// Baseline storage
typedef struct zap_baseline {
    zap_baseline_entry_t* entries;
    size_t                      count;
    size_t                      capacity;
} zap_baseline_t;

// Comparison result for a single benchmark
typedef struct zap_comparison {
    const char*         name;
    double              old_mean;
    double              new_mean;
    double              change_pct;     // Percentage change (negative = faster)
    zap_change_t  change;
    bool                significant;    // Statistically significant?
} zap_comparison_t;

// Color output mode
typedef enum zap_color_mode {
    ZAP_COLOR_AUTO = 0,   // Auto-detect TTY
    ZAP_COLOR_ALWAYS,     // Always use colors
    ZAP_COLOR_NEVER       // Never use colors
} zap_color_mode_t;

// Environment information
typedef struct zap_env {
    char cpu_model[128];
    char os_info[64];
    char compiler[64];
    int  cpu_cores;
    int  cpu_threads;
    // SIMD capabilities
    bool has_sse;
    bool has_sse2;
    bool has_sse3;
    bool has_ssse3;
    bool has_sse41;
    bool has_sse42;
    bool has_avx;
    bool has_avx2;
    bool has_avx512f;
    bool has_neon;
} zap_env_t;

// Maximum implementations in a comparison group
#ifndef ZAP_MAX_IMPLS
#define ZAP_MAX_IMPLS 8
#endif

// Result for a single implementation in a comparison
typedef struct zap_impl_result {
    char             name[64];
    zap_stats_t      stats;
    bool             valid;
} zap_impl_result_t;

// Comparison group configuration
typedef struct zap_compare_group {
    char                     name[128];
    zap_bench_config_t       config;
    size_t                   baseline_idx;
    bool                     header_printed;
    char                     tags[ZAP_MAX_TAGS][32];
    size_t                   tag_count;
} zap_compare_group_t;

// Comparison context for a single benchmark with multiple implementations
typedef struct zap_compare_ctx {
    zap_compare_group_t*     group;
    zap_benchmark_id_t       id;
    zap_impl_result_t        results[ZAP_MAX_IMPLS];
    size_t                   impl_count;
    void*                    input;
    size_t                   input_size;
    bool                     skipped;  // Set if filtered out or dry-run
} zap_compare_ctx_t;

// Maximum CLI tags for filtering
#ifndef ZAP_MAX_CLI_TAGS
#define ZAP_MAX_CLI_TAGS 16
#endif

// Global configuration
typedef struct zap_config {
    const char*          baseline_path;
    const char*          filter;         // Benchmark name filter pattern
    double               fail_threshold; /* Exit non-zero if regression > this % */
    bool                 save_baseline;
    bool                 compare;
    bool                 explicit_path;  // User specified a custom path
    bool                 json_output;    // Output results as JSON
    bool                 has_regression; /* Track if any benchmark regressed beyond threshold */
    zap_color_mode_t     color_mode;     // Color output mode
    bool                 dry_run;        // List benchmarks without running
    // CLI overrides for benchmark settings
    size_t               cli_samples;    // 0 = use default
    uint64_t             cli_warmup_ns;  // 0 = use default
    uint64_t             cli_time_ns;    // 0 = use default
    uint64_t             cli_min_iters;  // 0 = use default
    // Tag filtering
    char                 cli_tags[ZAP_MAX_CLI_TAGS][32];
    size_t               cli_tag_count;
    // Output verbosity flags (text only, JSON always shows all)
    bool                 show_env;        // Show environment info
    bool                 show_histogram;  // Show distribution histogram
    bool                 show_percentiles;/* Show p75/p90/p95/p99 */
    zap_baseline_t baseline;
    zap_env_t      env;             // System environment info
} zap_config_t;

// Global config instance
extern zap_config_t zap_g_config;

/* API DECLARATIONS */

// Timing functions
uint64_t zap_now_ns(void);

// Statistics functions
double zap_mean(const double* samples, size_t n);
double zap_median(double* samples, size_t n);
double zap_percentile(const double* sorted_samples, size_t n, double p);
double zap_std_dev(const double* samples, size_t n, double mean);
double zap_mad(double* samples, size_t n, double median);
void   zap_confidence_interval(const double* samples, size_t n,
                                     double mean, double std_dev,
                                     double* ci_lower, double* ci_upper);
void   zap_detect_outliers(const double* samples, size_t n,
                                 double median, double mad,
                                 size_t* low, size_t* high);
zap_stats_t zap_compute_stats(double* samples, size_t n);

// Black box - prevents compiler from optimizing away values
#define zap_black_box(val) zap__black_box_impl(&(val), sizeof(val))
void zap__black_box_impl(void* ptr, size_t size);

// Benchmark control
void zap_init(zap_t* c, const char* name);
void zap_cleanup(zap_t* c);
bool zap_loop_start(zap_t* c);
void zap_loop_end(zap_t* c);

// Reporting
void zap_report(const char* name, const zap_stats_t* stats);
void zap_report_json(const char* name, const zap_stats_t* stats,
                     const zap_comparison_t* cmp);
void zap_report_group_start(const char* name);
void zap_report_group_end(void);

// Runner
void zap_run_bench(const zap_bench_t* bench);
void zap_run_group(const zap_group_t* group);

// Runtime benchmark groups (criterion-rs style)
zap_runtime_group_t* zap_benchmark_group(const char* name);
void zap_group_measurement_time(zap_runtime_group_t* g, uint64_t ns);
void zap_group_warmup_time(zap_runtime_group_t* g, uint64_t ns);
void zap_group_sample_count(zap_runtime_group_t* g, size_t count);
void zap_group_setup(zap_runtime_group_t* g, zap_setup_fn setup);
void zap_group_teardown(zap_runtime_group_t* g, zap_teardown_fn teardown);
void zap_group_tag(zap_runtime_group_t* g, const char* tag);
void zap_group_finish(zap_runtime_group_t* g);

// Benchmark ID for parameterized benchmarks
zap_benchmark_id_t zap_benchmark_id(const char* label, int64_t param);
zap_benchmark_id_t zap_benchmark_id_str(const char* label, const char* param);

// Bencher functions (iter-style API)
void zap_bench_function(zap_runtime_group_t* g, const char* name,
                              zap_bencher_fn fn);
void zap_bench_with_input(zap_runtime_group_t* g,
                                zap_benchmark_id_t id,
                                void* input, size_t input_size,
                                zap_param_fn fn);

// iter() - run the benchmark closure
void zap_bencher_iter(zap_bencher_t* b, void (*fn)(void));
void zap_bencher_iter_custom(zap_bencher_t* b,
                                   void (*setup)(void*),
                                   void (*routine)(void*),
                                   void (*teardown)(void*),
                                   void* user_data);

// Throughput configuration
void zap_set_throughput_bytes(zap_t* c, size_t bytes_per_iter);
void zap_set_throughput_elements(zap_t* c, size_t elements_per_iter);
void zap_bencher_set_throughput_bytes(zap_bencher_t* b, size_t bytes_per_iter);
void zap_bencher_set_throughput_elements(zap_bencher_t* b, size_t elements_per_iter);

// Baseline management
void zap_baseline_init(zap_baseline_t* b);
void zap_baseline_free(zap_baseline_t* b);
void zap_baseline_add(zap_baseline_t* b, const char* name,
                            const zap_stats_t* stats);
const zap_baseline_entry_t* zap_baseline_find(
    const zap_baseline_t* b, const char* name);
bool zap_baseline_save(const zap_baseline_t* b, const char* path);
bool zap_baseline_load(zap_baseline_t* b, const char* path);

// Comparison
zap_comparison_t zap_compare(const zap_baseline_entry_t* baseline,
                                         const zap_stats_t* current);
void zap_report_comparison(const char* name, const zap_stats_t* stats,
                                 const zap_comparison_t* cmp);

/* CLI argument parsing */
void zap_parse_args(int argc, char** argv);

// Filter matching
bool zap_matches_filter(const char* name, const char* pattern);
bool zap_group_matches_tags(const zap_runtime_group_t* g);

// Status messages
void zap_status_warmup(const char* name);
void zap_status_measuring(const char* name);
void zap_status_clear(void);

// Environment detection
void zap_env_detect(zap_env_t* env);
void zap_env_print(const zap_env_t* env);
void zap_env_print_json(const zap_env_t* env);

// Comparison API - compare multiple implementations
zap_compare_group_t* zap_compare_group(const char* name);
void zap_compare_set_baseline(zap_compare_group_t* g, size_t idx);
void zap_compare_tag(zap_compare_group_t* g, const char* tag);
zap_compare_ctx_t* zap_compare_begin(zap_compare_group_t* g,
                                     zap_benchmark_id_t id,
                                     void* input, size_t input_size);
void zap_compare_impl(zap_compare_ctx_t* ctx, const char* name, zap_param_fn fn);
void zap_compare_end(zap_compare_ctx_t* ctx);
void zap_compare_group_finish(zap_compare_group_t* g);

/* MACROS */

/*
 * ZAP_LOOP - Main benchmarking loop
 * Usage:
 *   ZAP_LOOP(c) {
 *       // code to benchmark
 *   }
 */
#define ZAP_LOOP(c) \
    while (zap_loop_start(c)) \
        for (uint64_t _crit_i = 0; \
             _crit_i < (c)->iterations; \
             ++_crit_i)

/*
 * After the loop body, we need to call zap_loop_end.
 * This is handled by wrapping in a do-while with cleanup.
 * Actually, let's use a different approach with a for-loop wrapper.
 */
#undef ZAP_LOOP
#define ZAP_LOOP(c) \
    for (int _crit_done = 0; !_crit_done; ) \
        for (; zap_loop_start(c); _crit_done = 1, zap_loop_end(c)) \
            for (uint64_t _crit_i = 0; _crit_i < (c)->iterations; ++_crit_i)

/*
 * ZAP_ITER - Single expression benchmark (like b.iter(|| expr))
 * Usage:
 *   void bench_example(zap_bencher_t* b) {
 *       int data = 42;
 *       ZAP_ITER(b, {
 *           result = expensive_operation(data);
 *           zap_black_box(result);
 *       });
 *   }
 */
#define ZAP_ITER(b, block) \
    do { \
        zap_t* _c = (b)->state; \
        ZAP_LOOP(_c) { \
            block \
        } \
    } while (0)

/*
 * Duration helper macros (convert to nanoseconds)
 */
#define ZAP_SECONDS(s)      ((uint64_t)(s) * 1000000000ULL)
#define ZAP_MILLIS(ms)      ((uint64_t)(ms) * 1000000ULL)
#define ZAP_MICROS(us)      ((uint64_t)(us) * 1000ULL)

/*
 * ZAP_GROUP - Define a benchmark group
 * Usage:
 *   ZAP_GROUP(group_name, bench1, bench2, bench3);
 */
#define ZAP_GROUP(grpname, ...) \
    static const zap_bench_t grpname##_benches[] = { \
        ZAP__EXPAND_BENCHES(__VA_ARGS__) \
    }; \
    static const zap_group_t grpname = { \
        #grpname, \
        grpname##_benches, \
        sizeof(grpname##_benches) / sizeof(grpname##_benches[0]) \
    }

// Helper macro to expand benchmark function list
#define ZAP__EXPAND_BENCHES(...) \
    ZAP__MAP(ZAP__BENCH_ENTRY, __VA_ARGS__)

#define ZAP__BENCH_ENTRY(fn) { #fn, fn },

// Macro mapping utilities
#define ZAP__MAP(macro, ...) \
    ZAP__MAP_(__VA_ARGS__, \
        ZAP__MAP_16, ZAP__MAP_15, ZAP__MAP_14, \
        ZAP__MAP_13, ZAP__MAP_12, ZAP__MAP_11, \
        ZAP__MAP_10, ZAP__MAP_9,  ZAP__MAP_8, \
        ZAP__MAP_7,  ZAP__MAP_6,  ZAP__MAP_5, \
        ZAP__MAP_4,  ZAP__MAP_3,  ZAP__MAP_2, \
        ZAP__MAP_1)(macro, __VA_ARGS__)

#define ZAP__MAP_(...) ZAP__MAP_N(__VA_ARGS__)
#define ZAP__MAP_N(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15,_16,N,...) N

#define ZAP__MAP_1(m,a)  m(a)
#define ZAP__MAP_2(m,a,...)  m(a) ZAP__MAP_1(m,__VA_ARGS__)
#define ZAP__MAP_3(m,a,...)  m(a) ZAP__MAP_2(m,__VA_ARGS__)
#define ZAP__MAP_4(m,a,...)  m(a) ZAP__MAP_3(m,__VA_ARGS__)
#define ZAP__MAP_5(m,a,...)  m(a) ZAP__MAP_4(m,__VA_ARGS__)
#define ZAP__MAP_6(m,a,...)  m(a) ZAP__MAP_5(m,__VA_ARGS__)
#define ZAP__MAP_7(m,a,...)  m(a) ZAP__MAP_6(m,__VA_ARGS__)
#define ZAP__MAP_8(m,a,...)  m(a) ZAP__MAP_7(m,__VA_ARGS__)
#define ZAP__MAP_9(m,a,...)  m(a) ZAP__MAP_8(m,__VA_ARGS__)
#define ZAP__MAP_10(m,a,...) m(a) ZAP__MAP_9(m,__VA_ARGS__)
#define ZAP__MAP_11(m,a,...) m(a) ZAP__MAP_10(m,__VA_ARGS__)
#define ZAP__MAP_12(m,a,...) m(a) ZAP__MAP_11(m,__VA_ARGS__)
#define ZAP__MAP_13(m,a,...) m(a) ZAP__MAP_12(m,__VA_ARGS__)
#define ZAP__MAP_14(m,a,...) m(a) ZAP__MAP_13(m,__VA_ARGS__)
#define ZAP__MAP_15(m,a,...) m(a) ZAP__MAP_14(m,__VA_ARGS__)
#define ZAP__MAP_16(m,a,...) m(a) ZAP__MAP_15(m,__VA_ARGS__)

/*
 * ZAP_MAIN - Define main function that runs benchmark groups
 * Usage:
 *   ZAP_MAIN(group1, group2);
 *
 * Supports CLI arguments:
 *   --save-baseline [FILE]  Save results to baseline file
 *   --baseline [FILE]       Compare against baseline file
 *   --compare [FILE]        Alias for --baseline
 *   --json                  Output results as JSON
 *   --fail-threshold PCT    Exit with code 1 if regression > PCT%
 */
#define ZAP_MAIN(...) \
    int main(int argc, char** argv) { \
        zap_parse_args(argc, argv); \
        ZAP__MAP(ZAP__RUN_GROUP, __VA_ARGS__) \
        return zap_finalize(); \
    }

#define ZAP__RUN_GROUP(group) zap_run_group_internal(&group);

#ifdef __cplusplus
}
#endif

/* IMPLEMENTATION */

#ifdef ZAP_IMPLEMENTATION

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>  // For isatty()

/* POSIX timing */
#if defined(__APPLE__)
#include <mach/mach_time.h>
#include <sys/sysctl.h>
#else
#include <time.h>
#endif

/* CPUID for x86/x64 SIMD detection */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define ZAP_X86 1
#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#define ZAP_CPUID(leaf, a, b, c, d) __cpuid(leaf, a, b, c, d)
#define ZAP_CPUID_COUNT(leaf, sub, a, b, c, d) __cpuid_count(leaf, sub, a, b, c, d)
#elif defined(_MSC_VER)
#include <intrin.h>
#define ZAP_CPUID(leaf, a, b, c, d) do { \
    int regs[4]; __cpuid(regs, leaf); \
    a = regs[0]; b = regs[1]; c = regs[2]; d = regs[3]; \
} while(0)
#define ZAP_CPUID_COUNT(leaf, sub, a, b, c, d) do { \
    int regs[4]; __cpuidex(regs, leaf, sub); \
    a = regs[0]; b = regs[1]; c = regs[2]; d = regs[3]; \
} while(0)
#endif
#endif

/* ARM NEON detection */
#if defined(__aarch64__) || defined(_M_ARM64)
#define ZAP_ARM64 1
#endif

/* UTILITY MACROS */

#if defined(__GNUC__) || defined(__clang__)
#define ZAP_UNUSED __attribute__((unused))
#else
#define ZAP_UNUSED
#endif

/* ANSI COLOR CODES - Synthwave palette */

#define ZAP_COLOR_RESET   "\033[0m"
#define ZAP_COLOR_BOLD    "\033[1m"
#define ZAP_COLOR_DIM     "\033[2m"
// Synthwave neon/fluorescent colors (true color RGB)
#define ZAP_COLOR_GREEN   "\033[38;2;57;255;20m"   // Fluorescent green
#define ZAP_COLOR_YELLOW  "\033[38;2;255;170;0m"   // Neon orange/yellow
#define ZAP_COLOR_BLUE    "\033[38;2;0;191;255m"   // Electric blue
#define ZAP_COLOR_MAGENTA "\033[38;2;255;16;240m"  // Neon pink
#define ZAP_COLOR_CYAN    "\033[38;2;0;255;255m"   // Electric cyan
#define ZAP_COLOR_RED     "\033[38;2;255;0;63m"    // Neon red
#define ZAP_COLOR_PURPLE  "\033[38;2;191;0;255m"   // Neon purple

/* ENVIRONMENT DETECTION */

static void zap__detect_cpu_model(zap_env_t* env) {
#if defined(__APPLE__)
    size_t size = sizeof(env->cpu_model);
    if (sysctlbyname("machdep.cpu.brand_string", env->cpu_model, &size, NULL, 0) != 0) {
        strncpy(env->cpu_model, "Unknown", sizeof(env->cpu_model));
    }
#elif defined(__linux__)
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "model name", 10) == 0) {
                char* colon = strchr(line, ':');
                if (colon) {
                    colon++;
                    while (*colon == ' ' || *colon == '\t') colon++;
                    char* nl = strchr(colon, '\n');
                    if (nl) *nl = '\0';
                    strncpy(env->cpu_model, colon, sizeof(env->cpu_model) - 1);
                }
                break;
            }
        }
        fclose(f);
    } else {
        strncpy(env->cpu_model, "Unknown", sizeof(env->cpu_model));
    }
#else
    strncpy(env->cpu_model, "Unknown", sizeof(env->cpu_model));
#endif
    env->cpu_model[sizeof(env->cpu_model) - 1] = '\0';
}

static void zap__detect_cpu_cores(zap_env_t* env) {
#if defined(__APPLE__)
    size_t size = sizeof(env->cpu_cores);
    sysctlbyname("hw.physicalcpu", &env->cpu_cores, &size, NULL, 0);
    size = sizeof(env->cpu_threads);
    sysctlbyname("hw.logicalcpu", &env->cpu_threads, &size, NULL, 0);
#elif defined(_SC_NPROCESSORS_ONLN)
    env->cpu_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    env->cpu_cores = env->cpu_threads;  // Approximate
#else
    env->cpu_cores = 1;
    env->cpu_threads = 1;
#endif
}

static void zap__detect_os(zap_env_t* env) {
#if defined(__APPLE__)
    char version[32] = "";
    size_t size = sizeof(version);
    sysctlbyname("kern.osproductversion", version, &size, NULL, 0);
    snprintf(env->os_info, sizeof(env->os_info), "macOS %s", version);
#elif defined(__linux__)
    FILE* f = fopen("/etc/os-release", "r");
    if (f) {
        char line[128];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "PRETTY_NAME=", 12) == 0) {
                char* start = strchr(line, '"');
                if (start) {
                    start++;
                    char* end = strchr(start, '"');
                    if (end) *end = '\0';
                    strncpy(env->os_info, start, sizeof(env->os_info) - 1);
                }
                break;
            }
        }
        fclose(f);
    } else {
        strncpy(env->os_info, "Linux", sizeof(env->os_info));
    }
#elif defined(_WIN32)
    strncpy(env->os_info, "Windows", sizeof(env->os_info));
#else
    strncpy(env->os_info, "Unknown OS", sizeof(env->os_info));
#endif
    env->os_info[sizeof(env->os_info) - 1] = '\0';
}

static void zap__detect_compiler(zap_env_t* env) {
#if defined(__clang__)
    snprintf(env->compiler, sizeof(env->compiler), "Clang %d.%d.%d",
             __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
    snprintf(env->compiler, sizeof(env->compiler), "GCC %d.%d.%d",
             __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
    snprintf(env->compiler, sizeof(env->compiler), "MSVC %d", _MSC_VER);
#else
    strncpy(env->compiler, "Unknown compiler", sizeof(env->compiler));
#endif
    env->compiler[sizeof(env->compiler) - 1] = '\0';
}

static void zap__detect_simd(zap_env_t* env) {
    env->has_sse = false;
    env->has_sse2 = false;
    env->has_sse3 = false;
    env->has_ssse3 = false;
    env->has_sse41 = false;
    env->has_sse42 = false;
    env->has_avx = false;
    env->has_avx2 = false;
    env->has_avx512f = false;
    env->has_neon = false;

#if defined(ZAP_X86)
    unsigned int eax, ebx, ecx, edx;

    // Check CPUID availability and get feature flags
    ZAP_CPUID(1, eax, ebx, ecx, edx);

    env->has_sse    = (edx & (1 << 25)) != 0;
    env->has_sse2   = (edx & (1 << 26)) != 0;
    env->has_sse3   = (ecx & (1 << 0)) != 0;
    env->has_ssse3  = (ecx & (1 << 9)) != 0;
    env->has_sse41  = (ecx & (1 << 19)) != 0;
    env->has_sse42  = (ecx & (1 << 20)) != 0;
    env->has_avx    = (ecx & (1 << 28)) != 0;

    // Check for AVX2 and AVX-512 (need extended features)
    ZAP_CPUID(0, eax, ebx, ecx, edx);
    if (eax >= 7) {
        ZAP_CPUID_COUNT(7, 0, eax, ebx, ecx, edx);
        env->has_avx2    = (ebx & (1 << 5)) != 0;
        env->has_avx512f = (ebx & (1 << 16)) != 0;
    }
#endif

#if defined(ZAP_ARM64)
    // ARM64 always has NEON
    env->has_neon = true;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    env->has_neon = true;
#endif
}

void zap_env_detect(zap_env_t* env) {
    memset(env, 0, sizeof(*env));
    zap__detect_cpu_model(env);
    zap__detect_cpu_cores(env);
    zap__detect_os(env);
    zap__detect_compiler(env);
    zap__detect_simd(env);
}

/* TIMING IMPLEMENTATION */

uint64_t zap_now_ns(void) {
#if defined(__APPLE__)
    static mach_timebase_info_data_t timebase = {0};
    if (timebase.denom == 0) {
        mach_timebase_info(&timebase);
    }
    return mach_absolute_time() * timebase.numer / timebase.denom;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

/* STATISTICS IMPLEMENTATION */

static int zap__cmp_double(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

double zap_mean(const double* samples, size_t n) {
    if (n == 0) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += samples[i];
    }
    return sum / (double)n;
}

double zap_median(double* samples, size_t n) {
    if (n == 0) return 0.0;
    qsort(samples, n, sizeof(double), zap__cmp_double);
    if (n % 2 == 0) {
        return (samples[n/2 - 1] + samples[n/2]) / 2.0;
    }
    return samples[n/2];
}

double zap_percentile(const double* sorted_samples, size_t n, double p) {
    if (n == 0) return 0.0;
    if (n == 1) return sorted_samples[0];

    double rank = (p / 100.0) * (n - 1);
    size_t lower = (size_t)rank;
    size_t upper = lower + 1;
    if (upper >= n) upper = n - 1;

    double frac = rank - (double)lower;
    return sorted_samples[lower] * (1.0 - frac) + sorted_samples[upper] * frac;
}

double zap_std_dev(const double* samples, size_t n, double mean) {
    if (n < 2) return 0.0;
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = samples[i] - mean;
        sum_sq += diff * diff;
    }
    return sqrt(sum_sq / (double)(n - 1));
}

double zap_mad(double* samples, size_t n, double median) {
    if (n == 0) return 0.0;

    double* deviations = (double*)malloc(n * sizeof(double));
    if (!deviations) return 0.0;

    for (size_t i = 0; i < n; i++) {
        deviations[i] = fabs(samples[i] - median);
    }

    double result = zap_median(deviations, n);
    free(deviations);
    return result;
}

void zap_confidence_interval(const double* samples, size_t n,
                                   double mean, double std_dev,
                                   double* ci_lower, double* ci_upper) {
    (void)samples;  // CI computed from mean/std_dev directly
    // Using t-distribution approximation for 95% CI
    // For large n, t approaches 1.96
    double t = 1.96;
    if (n < 30) {
        // Rough approximation for small samples
        static const double t_values[] = {
            12.71, 4.30, 3.18, 2.78, 2.57,  // n = 2-6
            2.45, 2.36, 2.31, 2.26, 2.23,   // n = 7-11
            2.20, 2.18, 2.16, 2.14, 2.13,   // n = 12-16
            2.12, 2.11, 2.10, 2.09, 2.09,   // n = 17-21
            2.08, 2.07, 2.07, 2.06, 2.06,   // n = 22-26
            2.05, 2.05, 2.05                 // n = 27-29
        };
        if (n >= 2) {
            t = t_values[n - 2];
        }
    }

    double margin = t * std_dev / sqrt((double)n);
    *ci_lower = mean - margin;
    *ci_upper = mean + margin;
}

void zap_detect_outliers(const double* samples, size_t n,
                               double median, double mad,
                               size_t* low, size_t* high) {
    *low = 0;
    *high = 0;

    if (n == 0 || mad == 0.0) return;

    // Modified Z-score threshold (common value is 3.5)
    const double threshold = 3.5;
    // 0.6745 = 1/1.4826, where 1.4826 is consistency constant for normal dist

    for (size_t i = 0; i < n; i++) {
        double modified_z = 0.6745 * (samples[i] - median) / mad;
        if (modified_z < -threshold) {
            (*low)++;
        } else if (modified_z > threshold) {
            (*high)++;
        }
    }
}

zap_stats_t zap_compute_stats(double* samples, size_t n) {
    zap_stats_t stats = {0};
    if (n == 0) return stats;

    stats.sample_count = n;
    stats.samples = samples;  // Keep reference for histogram

    // Calculate min/max
    stats.min = samples[0];
    stats.max = samples[0];
    for (size_t i = 1; i < n; i++) {
        if (samples[i] < stats.min) stats.min = samples[i];
        if (samples[i] > stats.max) stats.max = samples[i];
    }

    // Need a copy for median since it sorts
    double* sorted = (double*)malloc(n * sizeof(double));
    if (!sorted) return stats;
    memcpy(sorted, samples, n * sizeof(double));

    stats.mean = zap_mean(samples, n);
    stats.median = zap_median(sorted, n);  // sorted is now sorted
    stats.std_dev = zap_std_dev(samples, n, stats.mean);

    // Calculate percentiles from sorted data
    stats.p75 = zap_percentile(sorted, n, 75.0);
    stats.p90 = zap_percentile(sorted, n, 90.0);
    stats.p95 = zap_percentile(sorted, n, 95.0);
    stats.p99 = zap_percentile(sorted, n, 99.0);

    // Need fresh copy for MAD calculation (it sorts internally)
    memcpy(sorted, samples, n * sizeof(double));
    stats.mad = zap_mad(sorted, n, stats.median);

    zap_confidence_interval(samples, n, stats.mean, stats.std_dev,
                                  &stats.ci_lower, &stats.ci_upper);

    zap_detect_outliers(samples, n, stats.median, stats.mad,
                              &stats.outliers_low, &stats.outliers_high);

    free(sorted);
    return stats;
}

/* BLACK BOX IMPLEMENTATION */

// Prevent compiler from optimizing away the value
void zap__black_box_impl(void* ptr, size_t size) {
    (void)size;
    __asm__ volatile("" : : "g"(ptr) : "memory");
}

/* STATUS MESSAGE IMPLEMENTATION */

static int zap__is_tty = -1;  // -1 = not checked yet

static int zap__check_tty(void) {
    if (zap__is_tty < 0) {
        zap__is_tty = isatty(STDOUT_FILENO);
    }
    return zap__is_tty;
}

// Check if colors should be used based on color_mode setting
static int zap__use_colors(void) {
    switch (zap_g_config.color_mode) {
        case ZAP_COLOR_ALWAYS: return 1;
        case ZAP_COLOR_NEVER:  return 0;
        default:               return zap__check_tty();
    }
}

// Conditional color codes - return empty string if colors disabled
static ZAP_UNUSED const char* zap__c_reset(void)   { return zap__use_colors() ? ZAP_COLOR_RESET : ""; }
static ZAP_UNUSED const char* zap__c_bold(void)    { return zap__use_colors() ? ZAP_COLOR_BOLD : ""; }
static ZAP_UNUSED const char* zap__c_dim(void)     { return zap__use_colors() ? ZAP_COLOR_DIM : ""; }
static ZAP_UNUSED const char* zap__c_green(void)   { return zap__use_colors() ? ZAP_COLOR_GREEN : ""; }
static ZAP_UNUSED const char* zap__c_yellow(void)  { return zap__use_colors() ? ZAP_COLOR_YELLOW : ""; }
static ZAP_UNUSED const char* zap__c_blue(void)    { return zap__use_colors() ? ZAP_COLOR_BLUE : ""; }
static ZAP_UNUSED const char* zap__c_cyan(void)    { return zap__use_colors() ? ZAP_COLOR_CYAN : ""; }
static ZAP_UNUSED const char* zap__c_red(void)     { return zap__use_colors() ? ZAP_COLOR_RED : ""; }
static ZAP_UNUSED const char* zap__c_magenta(void) { return zap__use_colors() ? ZAP_COLOR_MAGENTA : ""; }
static ZAP_UNUSED const char* zap__c_purple(void)  { return zap__use_colors() ? ZAP_COLOR_PURPLE : ""; }

/* ENVIRONMENT PRINT */

void zap_env_print(const zap_env_t* env) {
    printf("%s%sEnvironment:%s\n", zap__c_bold(), zap__c_magenta(), zap__c_reset());
    printf("  %sCPU:%s      %s%s%s\n",
           zap__c_dim(), zap__c_reset(),
           zap__c_cyan(), env->cpu_model, zap__c_reset());
    printf("  %sCores:%s    %s%d%s physical, %s%d%s logical\n",
           zap__c_dim(), zap__c_reset(),
           zap__c_cyan(), env->cpu_cores, zap__c_reset(),
           zap__c_cyan(), env->cpu_threads, zap__c_reset());
    printf("  %sOS:%s       %s\n",
           zap__c_dim(), zap__c_reset(), env->os_info);
    printf("  %sCompiler:%s %s\n",
           zap__c_dim(), zap__c_reset(), env->compiler);

    // Build SIMD string
    printf("  %sSIMD:%s     %s", zap__c_dim(), zap__c_reset(), zap__c_cyan());
    bool first = true;

#if defined(ZAP_X86)
    if (env->has_avx512f) { printf("%sAVX-512", first ? "" : ", "); first = false; }
    else if (env->has_avx2) { printf("%sAVX2", first ? "" : ", "); first = false; }
    else if (env->has_avx) { printf("%sAVX", first ? "" : ", "); first = false; }

    if (env->has_sse42) { printf("%sSSE4.2", first ? "" : ", "); first = false; }
    else if (env->has_sse41) { printf("%sSSE4.1", first ? "" : ", "); first = false; }
    else if (env->has_ssse3) { printf("%sSSSE3", first ? "" : ", "); first = false; }
    else if (env->has_sse3) { printf("%sSSE3", first ? "" : ", "); first = false; }
    else if (env->has_sse2) { printf("%sSSE2", first ? "" : ", "); first = false; }
    else if (env->has_sse) { printf("%sSSE", first ? "" : ", "); first = false; }
#endif

    if (env->has_neon) { printf("%sNEON", first ? "" : ", "); first = false; }

    if (first) printf("%snone detected", zap__c_yellow());
    printf("%s\n\n", zap__c_reset());
}

void zap_env_print_json(const zap_env_t* env) {
    printf("{\"type\":\"environment\"");
    printf(",\"cpu\":\"%s\"", env->cpu_model);
    printf(",\"cores\":%d", env->cpu_cores);
    printf(",\"threads\":%d", env->cpu_threads);
    printf(",\"os\":\"%s\"", env->os_info);
    printf(",\"compiler\":\"%s\"", env->compiler);

    printf(",\"simd\":[");
    bool first = true;
    if (env->has_sse) { printf("%s\"SSE\"", first ? "" : ","); first = false; }
    if (env->has_sse2) { printf("%s\"SSE2\"", first ? "" : ","); first = false; }
    if (env->has_sse3) { printf("%s\"SSE3\"", first ? "" : ","); first = false; }
    if (env->has_ssse3) { printf("%s\"SSSE3\"", first ? "" : ","); first = false; }
    if (env->has_sse41) { printf("%s\"SSE4.1\"", first ? "" : ","); first = false; }
    if (env->has_sse42) { printf("%s\"SSE4.2\"", first ? "" : ","); first = false; }
    if (env->has_avx) { printf("%s\"AVX\"", first ? "" : ","); first = false; }
    if (env->has_avx2) { printf("%s\"AVX2\"", first ? "" : ","); first = false; }
    if (env->has_avx512f) { printf("%s\"AVX-512\"", first ? "" : ","); first = false; }
    if (env->has_neon) { printf("%s\"NEON\"", first ? "" : ","); first = false; }
    printf("]");

    printf("}\n");
    fflush(stdout);
}

/* STATUS MESSAGE IMPLEMENTATION */

void zap_status_warmup(const char* name) {
    if (zap_g_config.json_output) return;  // No status in JSON mode
    if (zap__check_tty()) {
        // TTY: overwrite in place
        printf("\r\033[K%s  Warming up %s%s%s%s...%s",
               zap__c_dim(), zap__c_reset(),
               zap__c_magenta(), name, zap__c_reset(),
               zap__c_reset());
    } else {
        // Non-TTY: simple line
        printf("  Warming up %s...\n", name);
    }
    fflush(stdout);
}

void zap_status_measuring(const char* name) {
    if (zap_g_config.json_output) return;  // No status in JSON mode
    if (zap__check_tty()) {
        // TTY: overwrite in place
        printf("\r\033[K%s  Measuring  %s%s%s%s...%s",
               zap__c_dim(), zap__c_reset(),
               zap__c_magenta(), name, zap__c_reset(),
               zap__c_reset());
    } else {
        // Non-TTY: simple line
        printf("  Measuring  %s...\n", name);
    }
    fflush(stdout);
}

void zap_status_clear(void) {
    if (zap__check_tty()) {
        printf("\r\033[K");
        fflush(stdout);
    }
    // Non-TTY: nothing to clear, lines already printed
}

/* BENCHMARK CONTROL IMPLEMENTATION */

void zap_init(zap_t* c, const char* name) {
    memset(c, 0, sizeof(*c));
    c->name = name;
    c->iterations = 1;
    // Set default config, apply CLI overrides if specified
    c->config.warmup_time_ns = zap_g_config.cli_warmup_ns > 0
        ? zap_g_config.cli_warmup_ns : ZAP_DEFAULT_WARMUP_TIME_NS;
    c->config.measurement_time_ns = zap_g_config.cli_time_ns > 0
        ? zap_g_config.cli_time_ns : ZAP_DEFAULT_MEASUREMENT_TIME_NS;
    c->config.sample_count = zap_g_config.cli_samples > 0
        ? zap_g_config.cli_samples : ZAP_DEFAULT_SAMPLE_COUNT;
    if (zap_g_config.cli_min_iters > 0) {
        c->iterations = zap_g_config.cli_min_iters;
    }
    c->sample_capacity = c->config.sample_count;
    c->samples = (double*)malloc(c->sample_capacity * sizeof(double));
}

void zap_cleanup(zap_t* c) {
    free(c->samples);
    c->samples = NULL;
}

bool zap_loop_start(zap_t* c) {
    if (!c->warmup_complete) {
        // Warmup phase: run for warmup time while calibrating iterations
        uint64_t now = zap_now_ns();

        if (c->start_time == 0) {
            // First warmup iteration - print status
            zap_status_warmup(c->name);
            c->start_time = now;
            c->current_iter = now;
            return true;
        }

        // Measure time for previous warmup iteration batch
        uint64_t batch_elapsed = now - c->current_iter;
        uint64_t total_elapsed = now - c->start_time;

        // Calibrate: target 1ms per iteration batch
        if (batch_elapsed > 0 && batch_elapsed < 1000000) {
            // Scale up iterations to get closer to 1ms
            uint64_t factor = 1000000 / batch_elapsed;
            if (factor > 1) {
                c->iterations *= factor;
                if (c->iterations > 1000000000ULL) {
                    c->iterations = 1000000000ULL;
                }
            } else {
                c->iterations *= 2;
            }
        } else if (batch_elapsed > 10000000) {
            // Way too long, scale down
            c->iterations = (c->iterations > 2) ? c->iterations / 2 : 1;
        }

        if (total_elapsed >= c->config.warmup_time_ns) {
            // Warmup complete, switch to measurement
            c->warmup_complete = true;
            c->start_time = 0;
            c->measuring = false;
            c->status_printed = false;  // Reset for measuring status
        }

        c->current_iter = now;
        return true;
    }

    // Measurement phase
    if (c->sample_count >= c->sample_capacity) {
        return false;  // Done collecting samples
    }

    // Check if we've exceeded measurement time
    uint64_t now = zap_now_ns();
    if (c->start_time == 0) {
        // First measurement iteration - print status
        zap_status_measuring(c->name);
        c->start_time = now;
    } else {
        uint64_t elapsed = now - c->start_time;
        if (elapsed >= c->config.measurement_time_ns &&
            c->sample_count >= 10) {
            return false;  // Time's up and we have enough samples
        }
    }

    c->measuring = true;
    c->current_iter = zap_now_ns();
    return true;
}

void zap_loop_end(zap_t* c) {
    if (!c->measuring || !c->warmup_complete) return;

    uint64_t end = zap_now_ns();
    uint64_t elapsed = end - c->current_iter;

    // Store sample (time per iteration in nanoseconds)
    double time_per_iter = (double)elapsed / (double)c->iterations;

    if (c->sample_count < c->sample_capacity) {
        c->samples[c->sample_count++] = time_per_iter;
    }

    // Fine-tune iterations if samples are too short/long
    if (elapsed < 500000) {  // Less than 0.5ms
        c->iterations = c->iterations * 2;
        if (c->iterations > 1000000000ULL) {
            c->iterations = 1000000000ULL;
        }
    }

    c->measuring = false;
}

/* REPORTING IMPLEMENTATION */

// Greek mu (μ) for microseconds: UTF-8 = \316\274
static void zap__format_time(double ns, char* buf, size_t bufsize) {
    if (ns >= 1e9) {
        snprintf(buf, bufsize, "%.3f s", ns / 1e9);
    } else if (ns >= 1e6) {
        snprintf(buf, bufsize, "%.3f ms", ns / 1e6);
    } else if (ns >= 1e3) {
        snprintf(buf, bufsize, "%.3f \316\274s", ns / 1e3);
    } else {
        snprintf(buf, bufsize, "%.3f ns", ns);
    }
}

static void zap__format_time_short(double ns, char* buf, size_t bufsize) {
    if (ns >= 1e9) {
        snprintf(buf, bufsize, "%.2f s", ns / 1e9);
    } else if (ns >= 1e6) {
        snprintf(buf, bufsize, "%.2f ms", ns / 1e6);
    } else if (ns >= 1e3) {
        snprintf(buf, bufsize, "%.2f \316\274s", ns / 1e3);
    } else {
        snprintf(buf, bufsize, "%.2f ns", ns);
    }
}

// Format throughput: bytes/sec or elements/sec
static void zap__format_throughput(double mean_ns, size_t value,
                                   zap_throughput_type_t type,
                                   char* buf, size_t bufsize) {
    if (type == ZAP_THROUGHPUT_NONE || value == 0 || mean_ns <= 0) {
        buf[0] = '\0';
        return;
    }

    /* Calculate per-second rate: value / (mean_ns / 1e9) = value * 1e9 / mean_ns */
    double per_sec = (double)value * 1e9 / mean_ns;

    if (type == ZAP_THROUGHPUT_BYTES) {
        // Format as human-readable bytes/sec
        if (per_sec >= 1e12) {
            snprintf(buf, bufsize, "%.2f TB/s", per_sec / 1e12);
        } else if (per_sec >= 1e9) {
            snprintf(buf, bufsize, "%.2f GB/s", per_sec / 1e9);
        } else if (per_sec >= 1e6) {
            snprintf(buf, bufsize, "%.2f MB/s", per_sec / 1e6);
        } else if (per_sec >= 1e3) {
            snprintf(buf, bufsize, "%.2f KB/s", per_sec / 1e3);
        } else {
            snprintf(buf, bufsize, "%.2f B/s", per_sec);
        }
    } else {
        // Format as elements/sec (ops/sec)
        if (per_sec >= 1e9) {
            snprintf(buf, bufsize, "%.2f Gops/s", per_sec / 1e9);
        } else if (per_sec >= 1e6) {
            snprintf(buf, bufsize, "%.2f Mops/s", per_sec / 1e6);
        } else if (per_sec >= 1e3) {
            snprintf(buf, bufsize, "%.2f Kops/s", per_sec / 1e3);
        } else {
            snprintf(buf, bufsize, "%.2f ops/s", per_sec);
        }
    }
}

// Unicode block characters for histogram (increasing height)
// ▁▂▃▄▅▆▇█
static const char* zap__blocks[] = {
    " ", "\342\226\201", "\342\226\202", "\342\226\203",
    "\342\226\204", "\342\226\205", "\342\226\206", "\342\226\207",
    "\342\226\210"
};

static void zap__print_histogram(const double* samples, size_t n,
                                       double min_val, double max_val) {
    if (n == 0 || max_val <= min_val) return;

    // Format time labels first to determine histogram width (Julia style)
    char min_buf[32], max_buf[32];
    zap__format_time_short(min_val, min_buf, sizeof(min_buf));
    zap__format_time_short(max_val, max_buf, sizeof(max_buf));

    int min_len = (int)strlen(min_buf);
    int max_len = (int)strlen(max_buf);

    // Julia: histwidth = 42 + lmaxtimewidth + rmaxtimewidth, bins = histwidth - 1
    #define HIST_MAX_BINS 80
    #define HIST_HEIGHT 2
    int hist_width = 42 + min_len + max_len;
    int num_bins = hist_width - 1;
    if (num_bins > HIST_MAX_BINS) num_bins = HIST_MAX_BINS;
    if (num_bins < 10) num_bins = 10;

    int bins[HIST_MAX_BINS] = {0};

    double range = max_val - min_val;
    if (range <= 0) return;

    // Simple binning - count samples in each bin (Julia style)
    double bin_width = range / num_bins;
    for (size_t i = 0; i < n; i++) {
        int bin = (int)((samples[i] - min_val) / bin_width);
        if (bin >= num_bins) bin = num_bins - 1;
        if (bin < 0) bin = 0;
        bins[bin]++;
    }

    // Find max bin count
    int max_count = 0;
    for (int i = 0; i < num_bins; i++) {
        if (bins[i] > max_count) max_count = bins[i];
    }

    if (max_count == 0) return;

    /*
     * Julia-style multi-row histogram:
     * Compute bar heights in range [1, HIST_HEIGHT * 8]
     * Zero bins get height 0 (space)
     * Non-zero bins get at least height 1
     */
    int barheights[HIST_MAX_BINS];
    int total_levels = HIST_HEIGHT * 8;

    for (int i = 0; i < num_bins; i++) {
        if (bins[i] == 0) {
            barheights[i] = 0;
        } else {
            // Scale to [1, total_levels]
            barheights[i] = 1 + (int)((double)(bins[i] - 1) / (max_count - 1) * (total_levels - 1) + 0.5);
            if (barheights[i] > total_levels) barheights[i] = total_levels;
            if (max_count == 1) barheights[i] = total_levels; /* Single max gets full height */
        }
    }

    // Render rows from top to bottom
    for (int row = HIST_HEIGHT; row >= 1; row--) {
        printf("  ");
        int row_base = (row - 1) * 8;  // Heights 0-7 for row 1, 8-15 for row 2, etc

        for (int i = 0; i < num_bins; i++) {
            int height_in_row = barheights[i] - row_base;
            if (height_in_row <= 0) {
                printf(" ");
            } else if (height_in_row >= 8) {
                printf("%s", zap__blocks[8]);  // Full block
            } else {
                printf("%s", zap__blocks[height_in_row]);
            }
        }
        printf("\n");
    }

    #undef HIST_HEIGHT
    #undef HIST_MAX_BINS

    // Center the label text
    int label_len = 28;  // "Histogram: frequency by time"
    int total_width = num_bins + 2;
    int padding = (total_width - min_len - max_len - label_len) / 2;
    if (padding < 1) padding = 1;

    printf("  %s", min_buf);
    for (int i = 0; i < padding; i++) printf(" ");
    printf("%sHistogram: frequency by time%s", zap__c_cyan(), zap__c_reset());
    for (int i = 0; i < padding; i++) printf(" ");
    printf("%s\n", max_buf);

    #undef HIST_BINS
}

void zap_report(const char* name, const zap_stats_t* stats) {
    // Clear any status message before printing results
    zap_status_clear();

    char median_buf[32], mean_buf[32], std_buf[32];
    char min_buf[32], max_buf[32];

    zap__format_time(stats->median, median_buf, sizeof(median_buf));
    zap__format_time(stats->mean, mean_buf, sizeof(mean_buf));
    zap__format_time(stats->std_dev, std_buf, sizeof(std_buf));
    zap__format_time(stats->min, min_buf, sizeof(min_buf));
    zap__format_time(stats->max, max_buf, sizeof(max_buf));

    // Header - hot pink for benchmark name
    printf("%s%s%s:%s\n", zap__c_bold(), zap__c_magenta(), name, zap__c_reset());

    // Sample info line with median
    printf("  %zu samples \303\227 %zu evals, median: %s%s%s\n",
           stats->sample_count, stats->iterations,
           zap__c_cyan(), median_buf, zap__c_reset());

    // Time (mean ± σ)
    printf("  %sTime  (mean \302\261 \317\203):%s  %s%s%s \302\261 %s\n",
           zap__c_dim(), zap__c_reset(),
           zap__c_bold(), mean_buf, zap__c_reset(), std_buf);

    // Range (min … max) using ellipsis character
    printf("  %sRange (min \342\200\246 max):%s  %s \342\200\246 %s\n",
           zap__c_dim(), zap__c_reset(), min_buf, max_buf);

    // Percentiles (only with --percentiles flag)
    if (zap_g_config.show_percentiles) {
        char p75_buf[32], p90_buf[32], p95_buf[32], p99_buf[32];
        zap__format_time(stats->p75, p75_buf, sizeof(p75_buf));
        zap__format_time(stats->p90, p90_buf, sizeof(p90_buf));
        zap__format_time(stats->p95, p95_buf, sizeof(p95_buf));
        zap__format_time(stats->p99, p99_buf, sizeof(p99_buf));
        printf("  %sPercentiles:%s       p75: %s, p90: %s, p95: %s, p99: %s\n",
               zap__c_dim(), zap__c_reset(),
               p75_buf, p90_buf, p95_buf, p99_buf);
    }

    // Throughput if set
    if (stats->throughput_type != ZAP_THROUGHPUT_NONE && stats->throughput_value > 0) {
        char tput_buf[32];
        zap__format_throughput(stats->mean, stats->throughput_value,
                               stats->throughput_type, tput_buf, sizeof(tput_buf));
        printf("  %sThroughput:%s        %s%s%s\n",
               zap__c_dim(), zap__c_reset(),
               zap__c_cyan(), tput_buf, zap__c_reset());
    }

    // Outliers if any
    size_t total_outliers = stats->outliers_low + stats->outliers_high;
    if (total_outliers > 0) {
        printf("  %sOutliers:%s %s%zu low, %zu high%s\n",
               zap__c_dim(), zap__c_reset(),
               zap__c_yellow(), stats->outliers_low, stats->outliers_high, zap__c_reset());
    }

    // Histogram (only with --histogram flag)
    if (zap_g_config.show_histogram && stats->samples && stats->sample_count > 1) {
        printf("\n");
        zap__print_histogram(stats->samples, stats->sample_count,
                                   stats->min, stats->max);
    }

    printf("\n");
}

void zap_report_group_start(const char* name) {
    if (zap_g_config.json_output) return;  // No group headers in JSON mode
    printf("%s%sRunning benchmark group:%s %s%s%s\n\n",
           zap__c_bold(), zap__c_purple(), zap__c_reset(),
           zap__c_purple(), name, zap__c_reset());
}

void zap_report_group_end(void) {
    printf("\n");
}

/* RUNNER IMPLEMENTATION */

void zap_run_bench(const zap_bench_t* bench) {
    zap_t c;
    zap_init(&c, bench->name);

    // Run the benchmark function
    bench->func(&c);

    // Warn if time limit was reached before collecting all samples
    if (c.sample_count < c.config.sample_count) {
        printf("%sWarning: time limit reached, collected %zu/%zu samples%s\n",
               zap__c_yellow(), c.sample_count, c.config.sample_count, zap__c_reset());
    }

    // Compute and report statistics
    zap_stats_t stats = zap_compute_stats(c.samples, c.sample_count);
    stats.iterations = c.iterations;
    zap_report(bench->name, &stats);

    zap_cleanup(&c);
}

void zap_run_group(const zap_group_t* group) {
    zap_report_group_start(group->name);

    for (size_t i = 0; i < group->count; i++) {
        zap_run_bench(&group->benches[i]);
    }

    zap_report_group_end();
}

/* RUNTIME BENCHMARK GROUPS */

static zap_runtime_group_t zap__current_group = {0};

static bool zap__setup_called = false;  // Track if setup has been called for current group

zap_runtime_group_t* zap_benchmark_group(const char* name) {
    zap_runtime_group_t* g = &zap__current_group;

    strncpy(g->name, name, sizeof(g->name) - 1);
    g->name[sizeof(g->name) - 1] = '\0';

    // Set defaults
    g->config.warmup_time_ns = ZAP_DEFAULT_WARMUP_TIME_NS;
    g->config.measurement_time_ns = ZAP_DEFAULT_MEASUREMENT_TIME_NS;
    g->config.sample_count = ZAP_DEFAULT_SAMPLE_COUNT;
    g->active = true;
    g->header_printed = false;  // Defer header until first matching benchmark
    g->setup = NULL;
    g->teardown = NULL;
    g->tag_count = 0;
    zap__setup_called = false;

    // Only print header immediately if no filter is set and no tag filter
    if (!zap_g_config.filter && !zap_g_config.dry_run && zap_g_config.cli_tag_count == 0) {
        zap_report_group_start(name);
        g->header_printed = true;
    }
    return g;
}

void zap_group_measurement_time(zap_runtime_group_t* g, uint64_t ns) {
    g->config.measurement_time_ns = ns;
}

void zap_group_warmup_time(zap_runtime_group_t* g, uint64_t ns) {
    g->config.warmup_time_ns = ns;
}

void zap_group_sample_count(zap_runtime_group_t* g, size_t count) {
    g->config.sample_count = count;
}

void zap_group_setup(zap_runtime_group_t* g, zap_setup_fn setup) {
    g->setup = setup;
}

void zap_group_teardown(zap_runtime_group_t* g, zap_teardown_fn teardown) {
    g->teardown = teardown;
}

void zap_group_tag(zap_runtime_group_t* g, const char* tag) {
    if (g->tag_count < ZAP_MAX_TAGS) {
        strncpy(g->tags[g->tag_count], tag, sizeof(g->tags[0]) - 1);
        g->tags[g->tag_count][sizeof(g->tags[0]) - 1] = '\0';
        g->tag_count++;
    }
}

void zap_group_finish(zap_runtime_group_t* g) {
    // Call teardown if set and not in dry run mode
    if (g->teardown && !zap_g_config.dry_run) {
        g->teardown();
    }
    g->active = false;
    // Only print group end if header was printed
    if (g->header_printed) {
        zap_report_group_end();
    }
}

/* BENCHMARK ID */

zap_benchmark_id_t zap_benchmark_id(const char* label, int64_t param) {
    zap_benchmark_id_t id;
    strncpy(id.label, label, sizeof(id.label) - 1);
    id.label[sizeof(id.label) - 1] = '\0';
    snprintf(id.param_str, sizeof(id.param_str), "%lld", (long long)param);
    return id;
}

zap_benchmark_id_t zap_benchmark_id_str(const char* label, const char* param) {
    zap_benchmark_id_t id;
    strncpy(id.label, label, sizeof(id.label) - 1);
    id.label[sizeof(id.label) - 1] = '\0';
    strncpy(id.param_str, param, sizeof(id.param_str) - 1);
    id.param_str[sizeof(id.param_str) - 1] = '\0';
    return id;
}

/* BENCHER FUNCTIONS */

static void zap__init_with_config(zap_t* c, const char* name,
                                        const zap_bench_config_t* config) {
    memset(c, 0, sizeof(*c));
    c->name = name;
    c->iterations = 1;
    // Apply CLI overrides over group config
    c->config = *config;
    if (zap_g_config.cli_warmup_ns > 0) {
        c->config.warmup_time_ns = zap_g_config.cli_warmup_ns;
    }
    if (zap_g_config.cli_time_ns > 0) {
        c->config.measurement_time_ns = zap_g_config.cli_time_ns;
    }
    if (zap_g_config.cli_samples > 0) {
        c->config.sample_count = zap_g_config.cli_samples;
    }
    if (zap_g_config.cli_min_iters > 0) {
        c->iterations = zap_g_config.cli_min_iters;
    }
    c->sample_capacity = c->config.sample_count;
    c->samples = (double*)malloc(c->sample_capacity * sizeof(double));
}

// Print benchmark in dry-run mode
static void zap__print_dry_run(const char* group_name, const char* bench_name) {
    if (zap_g_config.json_output) {
        printf("{\"type\":\"benchmark\",\"group\":\"%s\",\"name\":\"%s\"}\n",
               group_name ? group_name : "", bench_name);
    } else {
        if (group_name) {
            printf("  %s%s%s/%s%s%s\n",
                   zap__c_purple(), group_name, zap__c_reset(),
                   zap__c_magenta(), bench_name, zap__c_reset());
        } else {
            printf("  %s%s%s\n", zap__c_magenta(), bench_name, zap__c_reset());
        }
    }
}

static void zap__run_and_report(zap_t* c, const char* name) {
    // Warn if time limit was reached before collecting all samples
    if (!zap_g_config.json_output && c->sample_count < c->config.sample_count) {
        printf("%sWarning: time limit reached, collected %zu/%zu samples%s\n",
               zap__c_yellow(), c->sample_count, c->config.sample_count, zap__c_reset());
    }

    zap_stats_t stats = zap_compute_stats(c->samples, c->sample_count);
    stats.iterations = c->iterations;
    stats.throughput_type = c->throughput_type;
    stats.throughput_value = c->throughput_value;

    // Check if we should compare against baseline
    zap_comparison_t cmp = {0};
    const zap_baseline_entry_t* baseline = NULL;

    if (zap_g_config.compare) {
        baseline = zap_baseline_find(&zap_g_config.baseline, name);
        if (baseline) {
            cmp = zap_compare(baseline, &stats);

            // Track regression for --fail-threshold
            if (zap_g_config.fail_threshold > 0.0 &&
                cmp.change == ZAP_REGRESSED &&
                cmp.change_pct > zap_g_config.fail_threshold) {
                zap_g_config.has_regression = true;
            }
        }
    }

    // Output results
    if (zap_g_config.json_output) {
        zap_report_json(name, &stats, baseline ? &cmp : NULL);
    } else if (baseline) {
        zap_report_comparison(name, &stats, &cmp);
    } else if (zap_g_config.compare) {
        printf("%s(new)%s ", zap__c_yellow(), zap__c_reset());
        zap_report(name, &stats);
    } else {
        zap_report(name, &stats);
    }

    // Save to baseline if requested
    if (zap_g_config.save_baseline) {
        zap_baseline_add(&zap_g_config.baseline, name, &stats);
    }
}

void zap_bench_function(zap_runtime_group_t* g, const char* name,
                              zap_bencher_fn fn) {
    // Check filter before running
    if (!zap_matches_filter(name, zap_g_config.filter)) {
        return;
    }

    // Check tag filter
    if (!zap_group_matches_tags(g)) {
        return;
    }

    // Dry run mode: just print the benchmark name
    if (zap_g_config.dry_run) {
        zap__print_dry_run(g->name, name);
        return;
    }

    // Print deferred group header on first matching benchmark
    if (!g->header_printed) {
        zap_report_group_start(g->name);
        g->header_printed = true;
    }

    // Call setup on first matching benchmark
    if (g->setup && !zap__setup_called) {
        g->setup();
        zap__setup_called = true;
    }

    zap_t c;
    zap__init_with_config(&c, name, &g->config);

    zap_bencher_t b;
    b.state = &c;
    b.group = g;
    strncpy(b.full_name, name, sizeof(b.full_name) - 1);
    b.full_name[sizeof(b.full_name) - 1] = '\0';

    // Run the benchmark
    fn(&b);

    // Report results
    zap__run_and_report(&c, name);

    zap_cleanup(&c);
}

void zap_bench_with_input(zap_runtime_group_t* g,
                                zap_benchmark_id_t id,
                                void* input, size_t input_size,
                                zap_param_fn fn) {
    (void)input_size;  // Reserved for future use

    // Build full name: "label/param"
    char full_name[256];
    snprintf(full_name, sizeof(full_name), "%s/%s", id.label, id.param_str);

    // Check filter before running
    if (!zap_matches_filter(full_name, zap_g_config.filter)) {
        return;
    }

    // Check tag filter
    if (!zap_group_matches_tags(g)) {
        return;
    }

    // Dry run mode: just print the benchmark name
    if (zap_g_config.dry_run) {
        zap__print_dry_run(g->name, full_name);
        return;
    }

    // Print deferred group header on first matching benchmark
    if (!g->header_printed) {
        zap_report_group_start(g->name);
        g->header_printed = true;
    }

    // Call setup on first matching benchmark
    if (g->setup && !zap__setup_called) {
        g->setup();
        zap__setup_called = true;
    }

    zap_t c;
    zap__init_with_config(&c, full_name, &g->config);

    zap_bencher_t b;
    b.state = &c;
    b.group = g;
    strncpy(b.full_name, full_name, sizeof(b.full_name) - 1);
    b.full_name[sizeof(b.full_name) - 1] = '\0';

    // Run the benchmark with input
    fn(&b, input);

    // Report results
    zap__run_and_report(&c, full_name);

    zap_cleanup(&c);
}

void zap_bencher_iter(zap_bencher_t* b, void (*fn)(void)) {
    zap_t* c = b->state;

    ZAP_LOOP(c) {
        fn();
    }
}

void zap_bencher_iter_custom(zap_bencher_t* b,
                                   void (*setup)(void*),
                                   void (*routine)(void*),
                                   void (*teardown)(void*),
                                   void* user_data) {
    zap_t* c = b->state;

    if (setup) setup(user_data);

    ZAP_LOOP(c) {
        routine(user_data);
    }

    if (teardown) teardown(user_data);
}

/* THROUGHPUT CONFIGURATION */

void zap_set_throughput_bytes(zap_t* c, size_t bytes_per_iter) {
    c->throughput_type = ZAP_THROUGHPUT_BYTES;
    c->throughput_value = bytes_per_iter;
}

void zap_set_throughput_elements(zap_t* c, size_t elements_per_iter) {
    c->throughput_type = ZAP_THROUGHPUT_ELEMENTS;
    c->throughput_value = elements_per_iter;
}

void zap_bencher_set_throughput_bytes(zap_bencher_t* b, size_t bytes_per_iter) {
    zap_set_throughput_bytes(b->state, bytes_per_iter);
}

void zap_bencher_set_throughput_elements(zap_bencher_t* b, size_t elements_per_iter) {
    zap_set_throughput_elements(b->state, elements_per_iter);
}

/* GLOBAL CONFIG */

zap_config_t zap_g_config = {0};

/* BASELINE MANAGEMENT IMPLEMENTATION */

void zap_baseline_init(zap_baseline_t* b) {
    memset(b, 0, sizeof(*b));
    b->capacity = 64;
    b->entries = (zap_baseline_entry_t*)malloc(
        b->capacity * sizeof(zap_baseline_entry_t));
}

void zap_baseline_free(zap_baseline_t* b) {
    free(b->entries);
    b->entries = NULL;
    b->count = 0;
    b->capacity = 0;
}

void zap_baseline_add(zap_baseline_t* b, const char* name,
                            const zap_stats_t* stats) {
    // Check if entry already exists - update it
    for (size_t i = 0; i < b->count; i++) {
        if (strcmp(b->entries[i].name, name) == 0) {
            zap_baseline_entry_t* e = &b->entries[i];
            e->mean = stats->mean;
            e->std_dev = stats->std_dev;
            e->ci_lower = stats->ci_lower;
            e->ci_upper = stats->ci_upper;
            return;
        }
    }

    // New entry - add it
    if (b->count >= b->capacity) {
        b->capacity *= 2;
        b->entries = (zap_baseline_entry_t*)realloc(
            b->entries, b->capacity * sizeof(zap_baseline_entry_t));
    }

    zap_baseline_entry_t* e = &b->entries[b->count++];
    strncpy(e->name, name, sizeof(e->name) - 1);
    e->name[sizeof(e->name) - 1] = '\0';
    e->mean = stats->mean;
    e->std_dev = stats->std_dev;
    e->ci_lower = stats->ci_lower;
    e->ci_upper = stats->ci_upper;
}

const zap_baseline_entry_t* zap_baseline_find(
    const zap_baseline_t* b, const char* name) {
    for (size_t i = 0; i < b->count; i++) {
        if (strcmp(b->entries[i].name, name) == 0) {
            return &b->entries[i];
        }
    }
    return NULL;
}

/*
 * Baseline file format (text):
 * Line 1: "zap-baseline v1"
 * Following lines: name|mean|std_dev|ci_lower|ci_upper
 */
bool zap_baseline_save(const zap_baseline_t* b, const char* path) {
    // Create parent directory if it doesn't exist
    char dir[256];
    strncpy(dir, path, sizeof(dir) - 1);
    dir[sizeof(dir) - 1] = '\0';
    char* last_slash = strrchr(dir, '/');
    if (last_slash) {
        *last_slash = '\0';
        mkdir(dir, 0755);  // Ignore error if already exists
    }

    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "Error: Cannot open '%s' for writing\n", path);
        return false;
    }

    fprintf(f, "zap-baseline v1\n");
    for (size_t i = 0; i < b->count; i++) {
        const zap_baseline_entry_t* e = &b->entries[i];
        fprintf(f, "%s|%.17g|%.17g|%.17g|%.17g\n",
                e->name, e->mean, e->std_dev, e->ci_lower, e->ci_upper);
    }

    fclose(f);
    // Message is printed by zap_finalize if needed
    return true;
}

bool zap_baseline_load(zap_baseline_t* b, const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        return false;  // File doesn't exist - not an error for comparison
    }

    char line[512];

    // Check header
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return false;
    }
    if (strncmp(line, "zap-baseline v1", 15) != 0) {
        fprintf(stderr, "Error: Invalid baseline file format\n");
        fclose(f);
        return false;
    }

    // Read entries
    while (fgets(line, sizeof(line), f)) {
        zap_baseline_entry_t e = {0};
        char* p = line;

        // Parse name
        char* sep = strchr(p, '|');
        if (!sep) continue;
        size_t name_len = (size_t)(sep - p);
        if (name_len >= sizeof(e.name)) name_len = sizeof(e.name) - 1;
        memcpy(e.name, p, name_len);
        e.name[name_len] = '\0';
        p = sep + 1;

        // Parse values
        if (sscanf(p, "%lf|%lf|%lf|%lf", &e.mean, &e.std_dev,
                   &e.ci_lower, &e.ci_upper) != 4) {
            continue;
        }

        // Add to baseline
        if (b->count >= b->capacity) {
            b->capacity = b->capacity ? b->capacity * 2 : 64;
            b->entries = (zap_baseline_entry_t*)realloc(
                b->entries, b->capacity * sizeof(zap_baseline_entry_t));
        }
        b->entries[b->count++] = e;
    }

    fclose(f);
    if (!zap_g_config.json_output) {
        printf("%sLoaded baseline:%s %s%s%s (%zu entries)\n\n",
               zap__c_purple(), zap__c_reset(),
               zap__c_cyan(), path, zap__c_reset(), b->count);
    }
    return true;
}

/* COMPARISON IMPLEMENTATION */

zap_comparison_t zap_compare(const zap_baseline_entry_t* baseline,
                                         const zap_stats_t* current) {
    zap_comparison_t cmp = {0};

    cmp.old_mean = baseline->mean;
    cmp.new_mean = current->mean;

    // Calculate percentage change (negative = improvement/faster)
    if (baseline->mean > 0) {
        cmp.change_pct = ((current->mean - baseline->mean) / baseline->mean) * 100.0;
    }

    /*
     * Determine if change is statistically significant by checking
     * if confidence intervals overlap. If they don't overlap,
     * the change is significant.
     */
    bool ci_overlap = !(current->ci_upper < baseline->ci_lower ||
                        current->ci_lower > baseline->ci_upper);

    cmp.significant = !ci_overlap;

    // Determine direction of change
    if (!cmp.significant || fabs(cmp.change_pct) < 1.0) {
        cmp.change = ZAP_NO_CHANGE;
    } else if (cmp.change_pct < 0) {
        cmp.change = ZAP_IMPROVED;
    } else {
        cmp.change = ZAP_REGRESSED;
    }

    return cmp;
}

void zap_report_comparison(const char* name, const zap_stats_t* stats,
                                 const zap_comparison_t* cmp) {
    // Clear any status message before printing results
    zap_status_clear();

    char median_buf[32], mean_buf[32], std_buf[32];
    char min_buf[32], max_buf[32], old_mean_buf[32];

    zap__format_time(stats->median, median_buf, sizeof(median_buf));
    zap__format_time(stats->mean, mean_buf, sizeof(mean_buf));
    zap__format_time(stats->std_dev, std_buf, sizeof(std_buf));
    zap__format_time(stats->min, min_buf, sizeof(min_buf));
    zap__format_time(stats->max, max_buf, sizeof(max_buf));
    zap__format_time(cmp->old_mean, old_mean_buf, sizeof(old_mean_buf));

    // Header - hot pink for benchmark name
    printf("%s%s%s:%s\n", zap__c_bold(), zap__c_magenta(), name, zap__c_reset());

    // Sample info line with median
    printf("  %zu samples \303\227 %zu evals, median: %s%s%s\n",
           stats->sample_count, stats->iterations,
           zap__c_cyan(), median_buf, zap__c_reset());

    // Time (mean ± σ)
    printf("  %sTime  (mean \302\261 \317\203):%s  %s%s%s \302\261 %s\n",
           zap__c_dim(), zap__c_reset(),
           zap__c_bold(), mean_buf, zap__c_reset(), std_buf);

    // Range (min … max) using ellipsis character
    printf("  %sRange (min \342\200\246 max):%s  %s \342\200\246 %s\n",
           zap__c_dim(), zap__c_reset(), min_buf, max_buf);

    // Percentiles (only with --percentiles flag)
    if (zap_g_config.show_percentiles) {
        char p75_buf[32], p90_buf[32], p95_buf[32], p99_buf[32];
        zap__format_time(stats->p75, p75_buf, sizeof(p75_buf));
        zap__format_time(stats->p90, p90_buf, sizeof(p90_buf));
        zap__format_time(stats->p95, p95_buf, sizeof(p95_buf));
        zap__format_time(stats->p99, p99_buf, sizeof(p99_buf));
        printf("  %sPercentiles:%s       p75: %s, p90: %s, p95: %s, p99: %s\n",
               zap__c_dim(), zap__c_reset(),
               p75_buf, p90_buf, p95_buf, p99_buf);
    }

    // Throughput if set
    if (stats->throughput_type != ZAP_THROUGHPUT_NONE && stats->throughput_value > 0) {
        char tput_buf[32];
        zap__format_throughput(stats->mean, stats->throughput_value,
                               stats->throughput_type, tput_buf, sizeof(tput_buf));
        printf("  %sThroughput:%s        %s%s%s\n",
               zap__c_dim(), zap__c_reset(),
               zap__c_cyan(), tput_buf, zap__c_reset());
    }

    // Show comparison
    const char* change_color;
    const char* change_text;
    char sign = (cmp->change_pct >= 0) ? '+' : '-';
    double abs_pct = fabs(cmp->change_pct);

    switch (cmp->change) {
        case ZAP_IMPROVED:
            change_color = zap__c_green();
            change_text = "\342\206\223 faster";  // ↓ faster
            break;
        case ZAP_REGRESSED:
            change_color = zap__c_red();
            change_text = "\342\206\221 slower";  // ↑ slower
            break;
        default:
            change_color = zap__c_purple();
            change_text = "\342\211\210";  // ≈
            break;
    }

    printf("  %sBaseline:%s          %s%c%.2f%% %s%s (was %s)\n",
           zap__c_dim(), zap__c_reset(),
           change_color, sign, abs_pct, change_text, zap__c_reset(), old_mean_buf);

    // Outliers if any
    size_t total_outliers = stats->outliers_low + stats->outliers_high;
    if (total_outliers > 0) {
        printf("  %sOutliers:%s %s%zu low, %zu high%s\n",
               zap__c_dim(), zap__c_reset(),
               zap__c_yellow(), stats->outliers_low, stats->outliers_high, zap__c_reset());
    }

    // Histogram (only with --histogram flag)
    if (zap_g_config.show_histogram && stats->samples && stats->sample_count > 1) {
        printf("\n");
        zap__print_histogram(stats->samples, stats->sample_count,
                                   stats->min, stats->max);
    }

    printf("\n");
}

void zap_report_json(const char* name, const zap_stats_t* stats,
                     const zap_comparison_t* cmp) {
    // Clear any status message
    zap_status_clear();

    printf("{\"name\":\"%s\"", name);
    printf(",\"samples\":%zu", stats->sample_count);
    printf(",\"iterations\":%zu", stats->iterations);
    printf(",\"mean_ns\":%.6f", stats->mean);
    printf(",\"median_ns\":%.6f", stats->median);
    printf(",\"std_dev_ns\":%.6f", stats->std_dev);
    printf(",\"min_ns\":%.6f", stats->min);
    printf(",\"max_ns\":%.6f", stats->max);
    printf(",\"p75_ns\":%.6f", stats->p75);
    printf(",\"p90_ns\":%.6f", stats->p90);
    printf(",\"p95_ns\":%.6f", stats->p95);
    printf(",\"p99_ns\":%.6f", stats->p99);
    printf(",\"ci_lower_ns\":%.6f", stats->ci_lower);
    printf(",\"ci_upper_ns\":%.6f", stats->ci_upper);
    printf(",\"outliers_low\":%zu", stats->outliers_low);
    printf(",\"outliers_high\":%zu", stats->outliers_high);

    // Throughput if set
    if (stats->throughput_type != ZAP_THROUGHPUT_NONE && stats->throughput_value > 0) {
        double per_sec = (double)stats->throughput_value * 1e9 / stats->mean;
        printf(",\"throughput\":{");
        printf("\"type\":\"%s\"",
               stats->throughput_type == ZAP_THROUGHPUT_BYTES ? "bytes" : "elements");
        printf(",\"value_per_iter\":%zu", stats->throughput_value);
        printf(",\"per_second\":%.2f", per_sec);
        printf("}");
    }

    if (cmp) {
        printf(",\"baseline\":{");
        printf("\"old_mean_ns\":%.6f", cmp->old_mean);
        printf(",\"change_pct\":%.4f", cmp->change_pct);
        printf(",\"significant\":%s", cmp->significant ? "true" : "false");
        printf(",\"status\":\"%s\"",
               cmp->change == ZAP_IMPROVED ? "improved" :
               cmp->change == ZAP_REGRESSED ? "regressed" : "unchanged");
        printf("}");
    }

    printf("}\n");
    fflush(stdout);
}

/* FILTER MATCHING IMPLEMENTATION */

/*
 * Simple glob-style pattern matching:
 *   *  - matches zero or more characters
 *   ?  - matches exactly one character
 *   If no wildcards present, performs substring search
 */
static bool zap__glob_match(const char* pattern, const char* str) {
    const char* p = pattern;
    const char* s = str;
    const char* star_p = NULL;
    const char* star_s = NULL;

    while (*s) {
        if (*p == '*') {
            star_p = p++;
            star_s = s;
        } else if (*p == '?' || *p == *s) {
            p++;
            s++;
        } else if (star_p) {
            p = star_p + 1;
            s = ++star_s;
        } else {
            return false;
        }
    }

    while (*p == '*') p++;
    return *p == '\0';
}

bool zap_matches_filter(const char* name, const char* pattern) {
    if (!pattern || !*pattern) return true;  // No filter = match all
    if (!name) return false;

    // Check if pattern contains wildcards
    bool has_wildcard = false;
    for (const char* p = pattern; *p; p++) {
        if (*p == '*' || *p == '?') {
            has_wildcard = true;
            break;
        }
    }

    if (has_wildcard) {
        return zap__glob_match(pattern, name);
    }

    // No wildcards: do case-sensitive substring search
    return strstr(name, pattern) != NULL;
}

bool zap_group_matches_tags(const zap_runtime_group_t* g) {
    // No CLI tags = all groups match
    if (zap_g_config.cli_tag_count == 0) return true;

    // Check if any group tag matches any CLI tag
    for (size_t i = 0; i < zap_g_config.cli_tag_count; i++) {
        for (size_t j = 0; j < g->tag_count; j++) {
            if (strcmp(zap_g_config.cli_tags[i], g->tags[j]) == 0) {
                return true;
            }
        }
    }
    return false;
}

/* CLI ARGUMENT PARSING */

// Parse time duration string like "2s", "500ms", "100us" into nanoseconds
static uint64_t zap__parse_duration(const char* str) {
    char* end;
    double value = strtod(str, &end);
    if (end == str) return 0;  // No number found

    // Skip whitespace
    while (*end == ' ') end++;

    // Parse unit
    if (*end == '\0' || strcmp(end, "s") == 0 || strcmp(end, "sec") == 0) {
        return (uint64_t)(value * 1e9);
    } else if (strcmp(end, "ms") == 0) {
        return (uint64_t)(value * 1e6);
    } else if (strcmp(end, "us") == 0 || strcmp(end, "\316\274s") == 0) {
        return (uint64_t)(value * 1e3);
    } else if (strcmp(end, "ns") == 0) {
        return (uint64_t)value;
    } else if (strcmp(end, "m") == 0 || strcmp(end, "min") == 0) {
        return (uint64_t)(value * 60e9);
    }

    // Default: assume seconds
    return (uint64_t)(value * 1e9);
}

void zap_parse_args(int argc, char** argv) {
    // Default: auto-compare and auto-save to .zap/baseline
    const char* default_baseline = ".zap/baseline";
    zap_g_config.baseline_path = default_baseline;
    zap_g_config.filter = NULL;          // No filter by default
    zap_g_config.fail_threshold = 0.0;   // No threshold by default
    zap_g_config.save_baseline = true;   // Auto-save by default
    zap_g_config.compare = true;         // Auto-compare by default
    zap_g_config.json_output = false;    // Human-readable by default
    zap_g_config.has_regression = false;
    zap_g_config.color_mode = (zap_color_mode_t)ZAP_DEFAULT_COLOR_MODE;
    zap_g_config.dry_run = false;

    // Output verbosity from compile-time defaults
    zap_g_config.show_env = ZAP_DEFAULT_SHOW_ENV;
    zap_g_config.show_histogram = ZAP_DEFAULT_SHOW_HISTOGRAM;
    zap_g_config.show_percentiles = ZAP_DEFAULT_SHOW_PERCENTILES;

    // Measurement overrides from compile-time defaults
    zap_g_config.cli_samples = 0;        // 0 = use per-group default
    zap_g_config.cli_warmup_ns = 0;      // 0 = use per-group default
    zap_g_config.cli_time_ns = 0;        // 0 = use per-group default
    zap_g_config.cli_min_iters = ZAP_DEFAULT_MIN_ITERS;
    zap_g_config.cli_tag_count = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--filter") == 0 || strcmp(argv[i], "-f") == 0) {
            if (i + 1 < argc) {
                zap_g_config.filter = argv[++i];
            } else {
                fprintf(stderr, "Error: --filter requires a pattern argument\n");
                exit(1);
            }
        } else if (strcmp(argv[i], "--json") == 0) {
            zap_g_config.json_output = true;
        } else if (strcmp(argv[i], "--fail-threshold") == 0) {
            if (i + 1 < argc) {
                zap_g_config.fail_threshold = atof(argv[++i]);
            } else {
                fprintf(stderr, "Error: --fail-threshold requires a percentage value\n");
                exit(1);
            }
        } else if (strcmp(argv[i], "--save-baseline") == 0) {
            zap_g_config.save_baseline = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                zap_g_config.baseline_path = argv[++i];
                zap_g_config.explicit_path = true;
            }
        } else if (strcmp(argv[i], "--baseline") == 0 ||
                   strcmp(argv[i], "--compare") == 0) {
            zap_g_config.compare = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                zap_g_config.baseline_path = argv[++i];
                zap_g_config.explicit_path = true;
            }
        } else if (strcmp(argv[i], "--no-save") == 0) {
            zap_g_config.save_baseline = false;
        } else if (strcmp(argv[i], "--no-compare") == 0) {
            zap_g_config.compare = false;
        } else if (strncmp(argv[i], "--color", 7) == 0) {
            const char* mode = NULL;
            if (argv[i][7] == '=') {
                mode = &argv[i][8];
            } else if (i + 1 < argc && argv[i + 1][0] != '-') {
                mode = argv[++i];
            } else {
                mode = "always";  // --color without arg means always
            }
            if (strcmp(mode, "auto") == 0) {
                zap_g_config.color_mode = ZAP_COLOR_AUTO;
            } else if (strcmp(mode, "always") == 0 || strcmp(mode, "yes") == 0) {
                zap_g_config.color_mode = ZAP_COLOR_ALWAYS;
            } else if (strcmp(mode, "never") == 0 || strcmp(mode, "no") == 0) {
                zap_g_config.color_mode = ZAP_COLOR_NEVER;
            } else {
                fprintf(stderr, "Error: --color must be auto, always, or never\n");
                exit(1);
            }
        } else if (strcmp(argv[i], "--samples") == 0) {
            if (i + 1 < argc) {
                zap_g_config.cli_samples = (size_t)atoi(argv[++i]);
            } else {
                fprintf(stderr, "Error: --samples requires a number\n");
                exit(1);
            }
        } else if (strcmp(argv[i], "--warmup") == 0) {
            if (i + 1 < argc) {
                zap_g_config.cli_warmup_ns = zap__parse_duration(argv[++i]);
            } else {
                fprintf(stderr, "Error: --warmup requires a duration (e.g., 2s, 500ms)\n");
                exit(1);
            }
        } else if (strcmp(argv[i], "--time") == 0) {
            if (i + 1 < argc) {
                zap_g_config.cli_time_ns = zap__parse_duration(argv[++i]);
            } else {
                fprintf(stderr, "Error: --time requires a duration (e.g., 5s, 1000ms)\n");
                exit(1);
            }
        } else if (strcmp(argv[i], "--min-iters") == 0) {
            if (i + 1 < argc) {
                zap_g_config.cli_min_iters = (uint64_t)atoll(argv[++i]);
            } else {
                fprintf(stderr, "Error: --min-iters requires a number\n");
                exit(1);
            }
        } else if (strcmp(argv[i], "--dry-run") == 0 || strcmp(argv[i], "--list") == 0) {
            zap_g_config.dry_run = true;
        } else if (strcmp(argv[i], "--env") == 0) {
            zap_g_config.show_env = true;
        } else if (strcmp(argv[i], "--histogram") == 0) {
            zap_g_config.show_histogram = true;
        } else if (strcmp(argv[i], "--percentiles") == 0) {
            zap_g_config.show_percentiles = true;
        } else if (strcmp(argv[i], "--tag") == 0 || strcmp(argv[i], "-t") == 0) {
            if (i + 1 < argc) {
                if (zap_g_config.cli_tag_count < ZAP_MAX_CLI_TAGS) {
                    strncpy(zap_g_config.cli_tags[zap_g_config.cli_tag_count],
                            argv[++i], sizeof(zap_g_config.cli_tags[0]) - 1);
                    zap_g_config.cli_tags[zap_g_config.cli_tag_count]
                        [sizeof(zap_g_config.cli_tags[0]) - 1] = '\0';
                    zap_g_config.cli_tag_count++;
                } else {
                    fprintf(stderr, "Warning: Too many tags (max %d)\n", ZAP_MAX_CLI_TAGS);
                    i++;
                }
            } else {
                fprintf(stderr, "Error: --tag requires a tag name\n");
                exit(1);
            }
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Benchmark options:\n");
            printf("  -f, --filter PATTERN    Only run benchmarks matching PATTERN\n");
            printf("                          Supports * (any chars) and ? (single char)\n");
            printf("                          Without wildcards, matches substring\n");
            printf("  -t, --tag TAG           Only run benchmarks in groups with TAG\n");
            printf("                          Can be specified multiple times (OR logic)\n");
            printf("  --json                  Output results as JSON (one object per line)\n");
            printf("  --fail-threshold PCT    Exit with code 1 if regression exceeds PCT%%\n");
            printf("  --baseline [FILE]       Use specific baseline file (default: %s)\n",
                   default_baseline);
            printf("  --save-baseline [FILE]  Alias for --baseline\n");
            printf("  --compare [FILE]        Alias for --baseline\n");
            printf("  --no-save               Don't save results to baseline\n");
            printf("  --no-compare            Don't compare against baseline\n");
            printf("  --color=MODE            Color output: auto (default), always, never\n");
            printf("\nMeasurement options:\n");
            printf("  --samples N             Number of samples to collect (default: 100)\n");
            printf("  --warmup TIME           Warmup duration (default: 1s)\n");
            printf("  --time TIME             Measurement duration (default: 3s)\n");
            printf("  --min-iters N           Minimum iterations per sample\n");
            printf("                          TIME formats: 5s, 500ms, 100us, 1m\n");
            printf("\nOutput options:\n");
            printf("  --env                   Show environment info (CPU, OS, SIMD)\n");
            printf("  --histogram             Show distribution histograms\n");
            printf("  --percentiles           Show p75/p90/p95/p99 percentiles\n");
            printf("\nOther options:\n");
            printf("  --dry-run, --list       List benchmarks without running them\n");
            printf("  -h, --help              Show this help\n");
            printf("\nBy default, results are saved to and compared against '%s'\n",
                   default_baseline);
            printf("\nExamples:\n");
            printf("  --filter sort           Match benchmarks containing 'sort'\n");
            printf("  --filter 'sort*'        Match benchmarks starting with 'sort'\n");
            printf("  --tag slow              Only run benchmarks tagged 'slow'\n");
            printf("  -t unit -t fast         Run benchmarks tagged 'unit' OR 'fast'\n");
            printf("  --json                  Output JSON for CI integration\n");
            printf("  --fail-threshold 5      Fail CI if any benchmark regresses >5%%\n");
            printf("  --samples 50 --time 2s  Quick run with fewer samples\n");
            exit(0);
        }
    }

    // Skip baseline loading in dry run mode
    if (zap_g_config.dry_run) {
        if (!zap_g_config.json_output) {
            printf("%s%sBenchmarks:%s\n", zap__c_bold(), zap__c_magenta(), zap__c_reset());
        }
        return;
    }

    // Initialize baseline storage
    zap_baseline_init(&zap_g_config.baseline);

    // Try to load existing baseline for comparison
    if (zap_g_config.compare) {
        if (!zap_baseline_load(&zap_g_config.baseline,
                                     zap_g_config.baseline_path)) {
            // No baseline exists yet - first run
            if (!zap_g_config.explicit_path) {
                // Silent on first run with default baseline
                zap_g_config.compare = false;
            } else {
                printf("%sWarning: No baseline found at '%s', running without comparison%s\n\n",
                       zap__c_yellow(), zap_g_config.baseline_path, zap__c_reset());
                zap_g_config.compare = false;
            }
        }
    }

    // Detect and print environment info
    zap_env_detect(&zap_g_config.env);
    if (zap_g_config.json_output) {
        // JSON always includes environment
        zap_env_print_json(&zap_g_config.env);
    } else if (zap_g_config.show_env) {
        // Text only shows env with --env flag
        zap_env_print(&zap_g_config.env);
    }
}

/* UPDATED RUNNER WITH COMPARISON SUPPORT */

static const char* zap__current_group_name = NULL;  // Track current group for dry-run

static void zap_run_bench_internal(const zap_bench_t* bench) {
    // Check filter before running
    if (!zap_matches_filter(bench->name, zap_g_config.filter)) {
        return;
    }

    // Dry run mode: just print the benchmark name
    if (zap_g_config.dry_run) {
        zap__print_dry_run(zap__current_group_name, bench->name);
        return;
    }

    zap_t c;
    zap_init(&c, bench->name);

    // Run the benchmark function
    bench->func(&c);

    // Warn if time limit was reached before collecting all samples
    if (!zap_g_config.json_output && c.sample_count < c.config.sample_count) {
        printf("%sWarning: time limit reached, collected %zu/%zu samples%s\n",
               zap__c_yellow(), c.sample_count, c.config.sample_count, zap__c_reset());
    }

    // Compute statistics
    zap_stats_t stats = zap_compute_stats(c.samples, c.sample_count);
    stats.iterations = c.iterations;
    stats.throughput_type = c.throughput_type;
    stats.throughput_value = c.throughput_value;

    // Check if we should compare against baseline
    zap_comparison_t cmp = {0};
    const zap_baseline_entry_t* baseline = NULL;

    if (zap_g_config.compare) {
        baseline = zap_baseline_find(&zap_g_config.baseline, bench->name);
        if (baseline) {
            cmp = zap_compare(baseline, &stats);

            // Track regression for --fail-threshold
            if (zap_g_config.fail_threshold > 0.0 &&
                cmp.change == ZAP_REGRESSED &&
                cmp.change_pct > zap_g_config.fail_threshold) {
                zap_g_config.has_regression = true;
            }
        }
    }

    // Output results
    if (zap_g_config.json_output) {
        zap_report_json(bench->name, &stats, baseline ? &cmp : NULL);
    } else if (baseline) {
        zap_report_comparison(bench->name, &stats, &cmp);
    } else if (zap_g_config.compare) {
        printf("%s(new)%s ", zap__c_yellow(), zap__c_reset());
        zap_report(bench->name, &stats);
    } else {
        zap_report(bench->name, &stats);
    }

    // Save to baseline if requested
    if (zap_g_config.save_baseline) {
        zap_baseline_add(&zap_g_config.baseline, bench->name, &stats);
    }

    zap_cleanup(&c);
}

static void zap_run_group_internal(const zap_group_t* group) {
    // Check if any benchmarks match the filter before printing header
    size_t matching = 0;
    if (zap_g_config.filter) {
        for (size_t i = 0; i < group->count; i++) {
            if (zap_matches_filter(group->benches[i].name, zap_g_config.filter)) {
                matching++;
            }
        }
        if (matching == 0) return;  // Skip group entirely
    }

    // Set current group for dry-run output
    zap__current_group_name = group->name;

    if (!zap_g_config.dry_run) {
        zap_report_group_start(group->name);
    }

    for (size_t i = 0; i < group->count; i++) {
        zap_run_bench_internal(&group->benches[i]);
    }

    if (!zap_g_config.dry_run) {
        zap_report_group_end();
    }

    zap__current_group_name = NULL;
}

/* COMPARISON API IMPLEMENTATION */

static zap_compare_group_t zap__compare_group = {0};
static zap_compare_ctx_t zap__compare_ctx = {0};

zap_compare_group_t* zap_compare_group(const char* name) {
    zap_compare_group_t* g = &zap__compare_group;
    memset(g, 0, sizeof(*g));

    strncpy(g->name, name, sizeof(g->name) - 1);
    g->name[sizeof(g->name) - 1] = '\0';

    // Set defaults
    g->config.warmup_time_ns = ZAP_DEFAULT_WARMUP_TIME_NS;
    g->config.measurement_time_ns = ZAP_DEFAULT_MEASUREMENT_TIME_NS;
    g->config.sample_count = ZAP_DEFAULT_SAMPLE_COUNT;
    g->baseline_idx = 0;  // First implementation is baseline by default
    g->header_printed = false;
    g->tag_count = 0;

    // Print header for comparison group (unless filtering/dry-run)
    if (!zap_g_config.filter && !zap_g_config.dry_run && zap_g_config.cli_tag_count == 0) {
        if (!zap_g_config.json_output) {
            printf("%s%sRunning comparison group:%s %s%s%s\n\n",
                   zap__c_bold(), zap__c_purple(), zap__c_reset(),
                   zap__c_purple(), name, zap__c_reset());
        }
        g->header_printed = true;
    }

    return g;
}

void zap_compare_set_baseline(zap_compare_group_t* g, size_t idx) {
    g->baseline_idx = idx;
}

void zap_compare_tag(zap_compare_group_t* g, const char* tag) {
    if (g->tag_count < ZAP_MAX_TAGS) {
        strncpy(g->tags[g->tag_count], tag, sizeof(g->tags[0]) - 1);
        g->tags[g->tag_count][sizeof(g->tags[0]) - 1] = '\0';
        g->tag_count++;
    }
}

// Check if comparison group matches tag filter
static bool zap__compare_group_matches_tags(const zap_compare_group_t* g) {
    if (zap_g_config.cli_tag_count == 0) return true;

    for (size_t i = 0; i < zap_g_config.cli_tag_count; i++) {
        for (size_t j = 0; j < g->tag_count; j++) {
            if (strcmp(zap_g_config.cli_tags[i], g->tags[j]) == 0) {
                return true;
            }
        }
    }
    return false;
}

zap_compare_ctx_t* zap_compare_begin(zap_compare_group_t* g,
                                     zap_benchmark_id_t id,
                                     void* input, size_t input_size) {
    zap_compare_ctx_t* ctx = &zap__compare_ctx;
    memset(ctx, 0, sizeof(*ctx));

    ctx->group = g;
    ctx->id = id;
    ctx->input = input;
    ctx->input_size = input_size;
    ctx->impl_count = 0;
    ctx->skipped = false;

    // Build full name for filtering
    char full_name[256];
    snprintf(full_name, sizeof(full_name), "%s/%s", id.label, id.param_str);

    // Check filter
    if (!zap_matches_filter(full_name, zap_g_config.filter)) {
        ctx->skipped = true;
        return ctx;
    }

    // Check tag filter
    if (!zap__compare_group_matches_tags(g)) {
        ctx->skipped = true;
        return ctx;
    }

    // Dry run mode
    if (zap_g_config.dry_run) {
        if (zap_g_config.json_output) {
            printf("{\"type\":\"comparison\",\"group\":\"%s\",\"name\":\"%s\"}\n",
                   g->name, full_name);
        } else {
            printf("  %s%s%s/%s%s%s (comparison)\n",
                   zap__c_purple(), g->name, zap__c_reset(),
                   zap__c_magenta(), full_name, zap__c_reset());
        }
        ctx->skipped = true;
        return ctx;
    }

    // Print deferred group header on first matching benchmark
    if (!g->header_printed) {
        if (!zap_g_config.json_output) {
            printf("%s%sRunning comparison group:%s %s%s%s\n\n",
                   zap__c_bold(), zap__c_purple(), zap__c_reset(),
                   zap__c_purple(), g->name, zap__c_reset());
        }
        g->header_printed = true;
    }

    return ctx;
}

void zap_compare_impl(zap_compare_ctx_t* ctx, const char* name, zap_param_fn fn) {
    if (ctx->skipped) return;
    if (ctx->impl_count >= ZAP_MAX_IMPLS) return;

    zap_compare_group_t* g = ctx->group;
    zap_impl_result_t* result = &ctx->results[ctx->impl_count];

    // Store implementation name
    strncpy(result->name, name, sizeof(result->name) - 1);
    result->name[sizeof(result->name) - 1] = '\0';

    // Build full benchmark name: "label/param [impl_name]"
    char bench_name[384];
    snprintf(bench_name, sizeof(bench_name), "%s/%s [%s]",
             ctx->id.label, ctx->id.param_str, name);

    // Initialize benchmark state
    zap_t c;
    zap__init_with_config(&c, bench_name, &g->config);

    // Create bencher
    zap_bencher_t b;
    b.state = &c;
    b.group = NULL;  // No runtime group
    strncpy(b.full_name, bench_name, sizeof(b.full_name) - 1);
    b.full_name[sizeof(b.full_name) - 1] = '\0';

    // Run the benchmark
    fn(&b, ctx->input);

    // Warn if time limit was reached
    if (!zap_g_config.json_output && c.sample_count < c.config.sample_count) {
        printf("%sWarning: time limit reached, collected %zu/%zu samples%s\n",
               zap__c_yellow(), c.sample_count, c.config.sample_count, zap__c_reset());
    }

    // Compute stats
    result->stats = zap_compute_stats(c.samples, c.sample_count);
    result->stats.iterations = c.iterations;
    result->stats.throughput_type = c.throughput_type;
    result->stats.throughput_value = c.throughput_value;
    result->valid = true;

    ctx->impl_count++;

    zap_cleanup(&c);
}

void zap_compare_end(zap_compare_ctx_t* ctx) {
    if (ctx->skipped) return;
    if (ctx->impl_count == 0) return;

    zap_compare_group_t* g = ctx->group;
    size_t baseline_idx = g->baseline_idx;
    if (baseline_idx >= ctx->impl_count) {
        baseline_idx = 0;
    }

    // Build comparison name
    char cmp_name[256];
    snprintf(cmp_name, sizeof(cmp_name), "%s/%s", ctx->id.label, ctx->id.param_str);

    // Clear status line
    zap_status_clear();

    if (zap_g_config.json_output) {
        // JSON output for comparison
        printf("{\"type\":\"comparison\",\"name\":\"%s\"", cmp_name);
        printf(",\"baseline_idx\":%zu", baseline_idx);
        printf(",\"implementations\":[");

        for (size_t i = 0; i < ctx->impl_count; i++) {
            zap_impl_result_t* r = &ctx->results[i];
            if (!r->valid) continue;

            if (i > 0) printf(",");
            printf("{\"name\":\"%s\"", r->name);
            printf(",\"mean_ns\":%.6f", r->stats.mean);
            printf(",\"median_ns\":%.6f", r->stats.median);
            printf(",\"std_dev_ns\":%.6f", r->stats.std_dev);
            printf(",\"samples\":%zu", r->stats.sample_count);
            printf(",\"iterations\":%zu", r->stats.iterations);

            // Throughput if set
            if (r->stats.throughput_type != ZAP_THROUGHPUT_NONE && r->stats.throughput_value > 0) {
                double per_sec = (double)r->stats.throughput_value * 1e9 / r->stats.mean;
                printf(",\"throughput\":{\"type\":\"%s\",\"per_second\":%.2f}",
                       r->stats.throughput_type == ZAP_THROUGHPUT_BYTES ? "bytes" : "elements",
                       per_sec);
            }

            // Calculate speedup vs baseline
            if (i != baseline_idx && ctx->results[baseline_idx].valid) {
                double speedup = ctx->results[baseline_idx].stats.mean / r->stats.mean;
                printf(",\"vs_baseline\":{\"speedup\":%.4f}", speedup);
            }

            // Compare with previous run baseline
            char full_bench_name[384];
            snprintf(full_bench_name, sizeof(full_bench_name), "%s/%s [%s]",
                     ctx->id.label, ctx->id.param_str, r->name);
            const zap_baseline_entry_t* prev = zap_baseline_find(&zap_g_config.baseline, full_bench_name);
            if (prev && zap_g_config.compare) {
                zap_comparison_t cmp = zap_compare(prev, &r->stats);
                printf(",\"vs_previous\":{\"old_mean_ns\":%.6f,\"change_pct\":%.4f,\"status\":\"%s\"}",
                       cmp.old_mean, cmp.change_pct,
                       cmp.change == ZAP_IMPROVED ? "improved" :
                       cmp.change == ZAP_REGRESSED ? "regressed" : "unchanged");

                // Track regression
                if (zap_g_config.fail_threshold > 0.0 &&
                    cmp.change == ZAP_REGRESSED &&
                    cmp.change_pct > zap_g_config.fail_threshold) {
                    zap_g_config.has_regression = true;
                }
            }

            printf("}");

            // Save to baseline
            if (zap_g_config.save_baseline) {
                zap_baseline_add(&zap_g_config.baseline, full_bench_name, &r->stats);
            }
        }

        printf("]}\n");
        fflush(stdout);
    } else {
        // Text output for comparison
        printf("%s%s%s comparison:\n", zap__c_bold(), zap__c_magenta(), cmp_name);

        for (size_t i = 0; i < ctx->impl_count; i++) {
            zap_impl_result_t* r = &ctx->results[i];
            if (!r->valid) continue;

            bool is_baseline = (i == baseline_idx);

            char median_buf[32], mean_buf[32], std_buf[32];
            char min_buf[32], max_buf[32];

            zap__format_time(r->stats.median, median_buf, sizeof(median_buf));
            zap__format_time(r->stats.mean, mean_buf, sizeof(mean_buf));
            zap__format_time(r->stats.std_dev, std_buf, sizeof(std_buf));
            zap__format_time(r->stats.min, min_buf, sizeof(min_buf));
            zap__format_time(r->stats.max, max_buf, sizeof(max_buf));

            // Implementation name header
            if (is_baseline) {
                printf("  %s%s%s %s[baseline]%s:\n",
                       zap__c_cyan(), r->name, zap__c_reset(),
                       zap__c_dim(), zap__c_reset());
            } else {
                printf("  %s%s%s:\n", zap__c_cyan(), r->name, zap__c_reset());
            }

            // Sample info
            printf("    %zu samples \303\227 %zu evals, median: %s%s%s\n",
                   r->stats.sample_count, r->stats.iterations,
                   zap__c_cyan(), median_buf, zap__c_reset());

            // Time (mean ± σ)
            printf("    %sTime  (mean \302\261 \317\203):%s  %s%s%s \302\261 %s\n",
                   zap__c_dim(), zap__c_reset(),
                   zap__c_bold(), mean_buf, zap__c_reset(), std_buf);

            // Range (min … max)
            printf("    %sRange (min \342\200\246 max):%s  %s \342\200\246 %s\n",
                   zap__c_dim(), zap__c_reset(), min_buf, max_buf);

            // Throughput if set
            if (r->stats.throughput_type != ZAP_THROUGHPUT_NONE && r->stats.throughput_value > 0) {
                char tput_buf[32];
                zap__format_throughput(r->stats.mean, r->stats.throughput_value,
                                       r->stats.throughput_type, tput_buf, sizeof(tput_buf));
                printf("    %sThroughput:%s        %s%s%s\n",
                       zap__c_dim(), zap__c_reset(),
                       zap__c_cyan(), tput_buf, zap__c_reset());
            }

            // Speedup vs all other implementations
            for (size_t j = 0; j < ctx->impl_count; j++) {
                if (j == i || !ctx->results[j].valid) continue;

                double speedup = ctx->results[j].stats.mean / r->stats.mean;
                const char* other_name = ctx->results[j].name;

                // Calculate padding for alignment
                int name_len = (int)strlen(other_name);
                int padding = 12 - name_len;
                if (padding < 1) padding = 1;

                if (speedup >= 1.0) {
                    printf("    %svs %s:%s%*s%s%.2fx faster%s\n",
                           zap__c_dim(), other_name, zap__c_reset(),
                           padding, "",
                           zap__c_green(), speedup, zap__c_reset());
                } else {
                    double pct_slower = (1.0 / speedup - 1.0) * 100.0;
                    printf("    %svs %s:%s%*s%s%.2fx (%.0f%% slower)%s\n",
                           zap__c_dim(), other_name, zap__c_reset(),
                           padding, "",
                           zap__c_red(), speedup, pct_slower, zap__c_reset());
                }
            }

            // Compare with previous run
            char full_bench_name[384];
            snprintf(full_bench_name, sizeof(full_bench_name), "%s/%s [%s]",
                     ctx->id.label, ctx->id.param_str, r->name);
            const zap_baseline_entry_t* prev = zap_baseline_find(&zap_g_config.baseline, full_bench_name);
            if (prev && zap_g_config.compare) {
                zap_comparison_t cmp = zap_compare(prev, &r->stats);

                char old_mean_buf[32];
                zap__format_time(cmp.old_mean, old_mean_buf, sizeof(old_mean_buf));

                const char* change_color;
                const char* change_text;
                char sign = (cmp.change_pct >= 0) ? '+' : '-';
                double abs_pct = fabs(cmp.change_pct);

                switch (cmp.change) {
                    case ZAP_IMPROVED:
                        change_color = zap__c_green();
                        change_text = "\342\206\223 faster";
                        break;
                    case ZAP_REGRESSED:
                        change_color = zap__c_red();
                        change_text = "\342\206\221 slower";
                        break;
                    default:
                        change_color = zap__c_purple();
                        change_text = "\342\211\210";
                        break;
                }

                printf("    %svs previous:%s      %s%c%.2f%% %s%s (was %s)\n",
                       zap__c_dim(), zap__c_reset(),
                       change_color, sign, abs_pct, change_text, zap__c_reset(), old_mean_buf);

                // Track regression
                if (zap_g_config.fail_threshold > 0.0 &&
                    cmp.change == ZAP_REGRESSED &&
                    cmp.change_pct > zap_g_config.fail_threshold) {
                    zap_g_config.has_regression = true;
                }
            }

            // Save to baseline
            if (zap_g_config.save_baseline) {
                zap_baseline_add(&zap_g_config.baseline, full_bench_name, &r->stats);
            }

            printf("\n");
        }
    }
}

void zap_compare_group_finish(zap_compare_group_t* g) {
    if (g->header_printed && !zap_g_config.json_output) {
        printf("\n");
    }
    memset(g, 0, sizeof(*g));
}

static int zap_finalize(void) {
    // Nothing to finalize in dry run mode
    if (zap_g_config.dry_run) {
        return 0;
    }

    // Save baseline if requested
    if (zap_g_config.save_baseline && zap_g_config.baseline.count > 0) {
        if (zap_baseline_save(&zap_g_config.baseline,
                                    zap_g_config.baseline_path)) {
            // Only print message for explicit path, not auto-save
            if (!zap_g_config.json_output && zap_g_config.explicit_path) {
                printf("%sBaseline saved to:%s %s%s%s\n",
                       zap__c_purple(), zap__c_reset(),
                       zap__c_cyan(), zap_g_config.baseline_path, zap__c_reset());
            }
        }
    }

    // Check for regressions beyond threshold
    if (zap_g_config.has_regression) {
        if (!zap_g_config.json_output) {
            fprintf(stderr, "%sError: Benchmark regression exceeded threshold (%.1f%%)%s\n",
                    zap__c_red(), zap_g_config.fail_threshold, zap__c_reset());
        }
    }

    // Cleanup
    if (zap_g_config.baseline.entries) {
        zap_baseline_free(&zap_g_config.baseline);
    }

    return zap_g_config.has_regression ? 1 : 0;
}

#endif /* ZAP_IMPLEMENTATION */

#endif /* ZAP_H */

/*
 * MIT License
 *
 * Copyright (c) 2024
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
