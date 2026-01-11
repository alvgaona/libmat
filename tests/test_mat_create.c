#define MATDEF static inline
#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

void test_mat_mat_zeros(void) {
    TEST_BEGIN("mat_mat creates zero matrix");
    Mat *m = mat_mat(2, 3);
    Mat *expected = mat_zeros(2, 3);

    CHECK(mat_equals(m, expected));
    CHECK(m->rows == 2 && m->cols == 3);

    mat_free_mat(m);
    mat_free_mat(expected);
    TEST_END();
}

void test_mat_zeros(void) {
    TEST_BEGIN("mat_zeros all zeros");
    Mat *z = mat_zeros(3, 3);

    int all_zero = 1;
    for (size_t i = 0; i < 9; i++) {
        if (z->data[i] != 0) all_zero = 0;
    }
    CHECK(all_zero);

    mat_free_mat(z);
    TEST_END();
}

void test_mat_ones(void) {
    TEST_BEGIN("mat_ones all ones");
    Mat *o = mat_ones(2, 3);

    int all_one = 1;
    for (size_t i = 0; i < 6; i++) {
        if (o->data[i] != 1) all_one = 0;
    }
    CHECK(all_one);
    CHECK(o->rows == 2 && o->cols == 3);

    mat_free_mat(o);
    TEST_END();
}

void test_mat_eye_diagonal(void) {
    TEST_BEGIN("mat_eye diagonal ones");
    Mat *eye = mat_reye(3);
    Mat *expected = mat_from(3, 3, (mat_elem_t[]){
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    });

    CHECK(mat_equals(eye, expected));

    mat_free_mat(eye);
    mat_free_mat(expected);
    TEST_END();
}

void test_mat_eye_square(void) {
    TEST_BEGIN("mat_eye is square");
    Mat *eye = mat_reye(4);

    CHECK(eye->rows == 4 && eye->cols == 4);

    mat_free_mat(eye);
    TEST_END();
}

void test_mat_eye_mul_identity(void) {
    TEST_BEGIN("mat_eye * A = A");
    Mat *a = mat_from(3, 3, (mat_elem_t[]){1, 2, 3, 4, 5, 6, 7, 8, 9});
    Mat *eye = mat_reye(3);

    Mat *result = mat_rmul(eye, a);
    CHECK(mat_equals(result, a));

    mat_free_mat(a);
    mat_free_mat(eye);
    mat_free_mat(result);
    TEST_END();
}

int main(void) {
    printf("mat_create:\n");

    test_mat_mat_zeros();
    test_mat_zeros();
    test_mat_ones();
    test_mat_eye_diagonal();
    test_mat_eye_square();
    test_mat_eye_mul_identity();

    TEST_SUMMARY();
}
