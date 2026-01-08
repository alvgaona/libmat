#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

void test_mat_trace(void) {
    TEST_BEGIN("mat_trace");
    Mat *a = mat_from(3, 3, (mat_elem_t[]){
      1, 2, 3,
      4, 5, 6,
      7, 8, 9
    });

    mat_elem_t trace = mat_trace(a);
  
    CHECK(trace == 15);
  
    mat_free_mat(a);
    TEST_END();
}

void test_mat_trace_negative(void) {
  TEST_BEGIN("mat_trace negative");
    Mat *a = mat_from(3, 3, (mat_elem_t[]){
      -1, 2.0, 1.5,
      0, -1, 0.78,
      0.5, 0.1, -1
    });

    mat_elem_t trace = mat_trace(a);

    CHECK(trace == -3);

    mat_free_mat(a);
    TEST_END();
}

void test_mat_trace_1x1(void) {
    TEST_BEGIN("mat_trace 1x1");
    Mat *a = mat_from(1, 1, (mat_elem_t[]){42});

    mat_elem_t trace = mat_trace(a);

    CHECK(trace == 42);

    mat_free_mat(a);
    TEST_END();
}

void test_mat_trace_2x2(void) {
    TEST_BEGIN("mat_trace 2x2");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){
      5, 3,
      2, 7
    });

    mat_elem_t trace = mat_trace(a);

    CHECK(trace == 12);

    mat_free_mat(a);
    TEST_END();
}

void test_mat_trace_identity(void) {
    TEST_BEGIN("mat_trace identity");
    Mat *a = mat_eye(4);

    mat_elem_t trace = mat_trace(a);

    CHECK(trace == 4);

    mat_free_mat(a);
    TEST_END();
}

int main(void) {
    printf("mat_trace:\n");

    test_mat_trace();
    test_mat_trace_negative();
    test_mat_trace_1x1();
    test_mat_trace_2x2();
    test_mat_trace_identity();

    TEST_SUMMARY();
}
