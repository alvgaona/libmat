#define MATDEF static inline
#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

static void test_cross_i_x_j(void) {
    TEST_BEGIN("cross_i_x_j");

    Vec *v1 = mat_vnew({1.0f, 0.0f, 0.0f});
    Vec *v2 = mat_vnew({0.0f, 1.0f, 0.0f});
    Vec *out = mat_vec(3);

    mat_cross(out, v1, v2);

    // i x j = k
    Vec *expected = mat_vnew({0.0f, 0.0f, 1.0f});
    CHECK(mat_equals(out, expected));

    mat_free_mat(v1);
    mat_free_mat(v2);
    mat_free_mat(out);
    mat_free_mat(expected);

    TEST_END();
}

static void test_cross_j_x_i(void) {
    TEST_BEGIN("cross_j_x_i");

    // j x i = -k
    Vec *v1 = mat_vnew({0.0f, 1.0f, 0.0f});
    Vec *v2 = mat_vnew({1.0f, 0.0f, 0.0f});
    Vec *out = mat_vec(3);

    mat_cross(out, v1, v2);

    Vec *expected = mat_vnew({0.0f, 0.0f, -1.0f});
    CHECK(mat_equals(out, expected));

    mat_free_mat(v1);
    mat_free_mat(v2);
    mat_free_mat(out);
    mat_free_mat(expected);

    TEST_END();
}

static void test_cross_general(void) {
    TEST_BEGIN("cross_general");

    Vec *v1 = mat_vnew({1.0f, 2.0f, 3.0f});
    Vec *v2 = mat_vnew({4.0f, 5.0f, 6.0f});
    Vec *out = mat_vec(3);

    mat_cross(out, v1, v2);

    // (1,2,3) x (4,5,6) = (2*6-3*5, 3*4-1*6, 1*5-2*4) = (-3, 6, -3)
    Vec *expected = mat_vnew({-3.0f, 6.0f, -3.0f});
    CHECK(mat_equals(out, expected));

    mat_free_mat(v1);
    mat_free_mat(v2);
    mat_free_mat(out);
    mat_free_mat(expected);

    TEST_END();
}

static void test_outer_2x3(void) {
    TEST_BEGIN("outer_2x3");

    Vec *v1 = mat_vnew({1.0f, 2.0f});
    Vec *v2 = mat_vnew({3.0f, 4.0f, 5.0f});
    Mat *out = mat_mat(2, 3);

    mat_outer(out, v1, v2);

    // [1]   [3 4 5]   [3  4  5]
    // [2] *         = [6  8  10]
    Mat *expected = mat_new(3, {
        {3.0f, 4.0f, 5.0f},
        {6.0f, 8.0f, 10.0f}
    });

    CHECK(mat_equals(out, expected));

    mat_free_mat(v1);
    mat_free_mat(v2);
    mat_free_mat(out);
    mat_free_mat(expected);

    TEST_END();
}

static void test_outer_3x2(void) {
    TEST_BEGIN("outer_3x2");

    Vec *v1 = mat_vnew({1.0f, 2.0f, 3.0f});
    Vec *v2 = mat_vnew({4.0f, 5.0f});
    Mat *out = mat_mat(3, 2);

    mat_outer(out, v1, v2);

    Mat *expected = mat_new(2, {
        {4.0f, 5.0f},
        {8.0f, 10.0f},
        {12.0f, 15.0f}
    });

    CHECK(mat_equals(out, expected));

    mat_free_mat(v1);
    mat_free_mat(v2);
    mat_free_mat(out);
    mat_free_mat(expected);

    TEST_END();
}

int main(void) {
    test_cross_i_x_j();
    test_cross_j_x_i();
    test_cross_general();
    test_outer_2x3();
    test_outer_3x2();

    TEST_SUMMARY();
}
