#define MATDEF static inline
#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

void test_mat_at_basic(void) {
    TEST_BEGIN("mat_at basic access");
    Mat *a = mat_from(2, 3, (mat_elem_t[]){1, 2, 3, 4, 5, 6});

    CHECK(mat_at(a, 0, 0) == 1);
    CHECK(mat_at(a, 0, 1) == 2);
    CHECK(mat_at(a, 0, 2) == 3);
    CHECK(mat_at(a, 1, 0) == 4);
    CHECK(mat_at(a, 1, 1) == 5);
    CHECK(mat_at(a, 1, 2) == 6);

    mat_free_mat(a);
    TEST_END();
}

void test_mat_at_corners(void) {
    TEST_BEGIN("mat_at corners");
    Mat *a = mat_from(3, 3, (mat_elem_t[]){
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });

    CHECK(mat_at(a, 0, 0) == 1);  // top-left
    CHECK(mat_at(a, 0, 2) == 3);  // top-right
    CHECK(mat_at(a, 2, 0) == 7);  // bottom-left
    CHECK(mat_at(a, 2, 2) == 9);  // bottom-right

    mat_free_mat(a);
    TEST_END();
}

void test_mat_size_basic(void) {
    TEST_BEGIN("mat_size basic");
    Mat *a = mat_from(2, 3, (mat_elem_t[]){1, 2, 3, 4, 5, 6});

    MatSize size = mat_size(a);
    CHECK(size.x == 2 && size.y == 3);

    mat_free_mat(a);
    TEST_END();
}

void test_mat_size_square(void) {
    TEST_BEGIN("mat_size square");
    Mat *a = mat_reye(4);

    MatSize size = mat_size(a);
    CHECK(size.x == 4 && size.y == 4);

    mat_free_mat(a);
    TEST_END();
}

void test_mat_size_vector(void) {
    TEST_BEGIN("mat_size vector");
    Vec *v = mat_vec_from(5, (mat_elem_t[]){1, 2, 3, 4, 5});

    MatSize size = mat_size(v);
    CHECK(size.x == 5 && size.y == 1);

    mat_free_mat(v);
    TEST_END();
}

int main(void) {
    printf("mat_access:\n");

    test_mat_at_basic();
    test_mat_at_corners();
    test_mat_size_basic();
    test_mat_size_square();
    test_mat_size_vector();

    TEST_SUMMARY();
}
