#ifndef TEST_H
#define TEST_H

#include <stdio.h>
#include <math.h>

static int tests_passed = 0;
static int tests_failed = 0;
static int current_test_failed = 0;
static const char *current_test_name = NULL;

#define TEST_BEGIN(name) do { \
    current_test_name = name; \
    current_test_failed = 0; \
} while(0)

#define TEST_END() do { \
    if (current_test_failed) { \
        tests_failed++; \
    } else { \
        printf("  %s... PASSED\n", current_test_name); \
        tests_passed++; \
    } \
} while(0)

#define CHECK(cond) do { \
    if (!(cond)) { \
        if (!current_test_failed) { \
            printf("  %s... FAILED\n", current_test_name); \
            printf("    -> %s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); \
        } else { \
            printf("    -> %s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); \
        } \
        current_test_failed = 1; \
    } \
} while(0)

#define CHECK_FLOAT_EQ(a, b) do { \
    if (fabsf((a) - (b)) >= 1e-6f) { \
        if (!current_test_failed) { \
            printf("  %s... FAILED\n", current_test_name); \
            printf("    -> %s:%d: CHECK_FLOAT_EQ(%s, %s) [%g != %g]\n", \
                   __FILE__, __LINE__, #a, #b, (double)(a), (double)(b)); \
        } else { \
            printf("    -> %s:%d: CHECK_FLOAT_EQ(%s, %s) [%g != %g]\n", \
                   __FILE__, __LINE__, #a, #b, (double)(a), (double)(b)); \
        } \
        current_test_failed = 1; \
    } \
} while(0)

#define TEST_SUMMARY() do { \
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed); \
    return tests_failed > 0 ? 1 : 0; \
} while(0)

#endif // TEST_H
