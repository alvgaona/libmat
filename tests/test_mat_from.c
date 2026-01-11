#define MATDEF static inline
#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

void test_mat_from_basic(void) {
    TEST_BEGIN("mat_from basic");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});

    CHECK(a->rows == 2 && a->cols == 2);
    CHECK(a->data[0] == 1 && a->data[1] == 2);
    CHECK(a->data[2] == 3 && a->data[3] == 4);

    mat_free_mat(a);
    TEST_END();
}

void test_mat_from_rectangular(void) {
    TEST_BEGIN("mat_from rectangular");
    Mat *a = mat_from(2, 3, (mat_elem_t[]){1, 2, 3, 4, 5, 6});

    CHECK(a->rows == 2 && a->cols == 3);
    CHECK(mat_at(a, 0, 0) == 1 && mat_at(a, 0, 2) == 3);
    CHECK(mat_at(a, 1, 0) == 4 && mat_at(a, 1, 2) == 6);

    mat_free_mat(a);
    TEST_END();
}

void test_mat_vec_from_basic(void) {
    TEST_BEGIN("mat_vec_from basic");
    Vec *v = mat_vec_from(3, (mat_elem_t[]){1, 2, 3});

    CHECK(v->rows == 3 && v->cols == 1);
    CHECK(v->data[0] == 1 && v->data[1] == 2 && v->data[2] == 3);

    mat_free_mat(v);
    TEST_END();
}

void test_mat_init(void) {
    TEST_BEGIN("mat_init overwrites data");
    Mat *a = mat_mat(2, 2);
    mat_init(a, (mat_elem_t[]){5, 6, 7, 8});

    Mat *expected = mat_from(2, 2, (mat_elem_t[]){5, 6, 7, 8});
    CHECK(mat_equals(a, expected));

    mat_free_mat(a);
    mat_free_mat(expected);
    TEST_END();
}

void test_mat_from_single_element(void) {
    TEST_BEGIN("mat_from single element");
    Mat *a = mat_from(1, 1, (mat_elem_t[]){42});

    CHECK(a->rows == 1 && a->cols == 1);
    CHECK(a->data[0] == 42);

    mat_free_mat(a);
    TEST_END();
}

int main(void) {
    printf("mat_from:\n");

    test_mat_from_basic();
    test_mat_from_rectangular();
    test_mat_vec_from_basic();
    test_mat_init();
    test_mat_from_single_element();

    TEST_SUMMARY();
}
