#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

void test_mat_copy_dimensions(void) {
    TEST_BEGIN("mat_copy preserves dimensions");
    Mat *a = mat_from(2, 3, (mat_elem_t[]){1, 2, 3, 4, 5, 6});

    Mat *copy = mat_copy(a);
    CHECK(copy->rows == a->rows && copy->cols == a->cols);

    mat_free_mat(a);
    mat_free_mat(copy);
    TEST_END();
}

void test_mat_deep_copy_values(void) {
    TEST_BEGIN("mat_deep_copy copies values");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});

    Mat *copy = mat_deep_copy(a);
    CHECK(mat_equals(copy, a));

    mat_free_mat(a);
    mat_free_mat(copy);
    TEST_END();
}

void test_mat_deep_copy_independent(void) {
    TEST_BEGIN("mat_deep_copy is independent");
    Mat *a = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});
    Mat *original = mat_from(2, 2, (mat_elem_t[]){1, 2, 3, 4});

    Mat *copy = mat_deep_copy(a);
    // Modify original
    a->data[0] = 999;

    // Copy should be unchanged
    CHECK(mat_equals(copy, original));

    mat_free_mat(a);
    mat_free_mat(copy);
    mat_free_mat(original);
    TEST_END();
}

void test_mat_deep_copy_rectangular(void) {
    TEST_BEGIN("mat_deep_copy rectangular");
    Mat *a = mat_from(2, 3, (mat_elem_t[]){1, 2, 3, 4, 5, 6});

    Mat *copy = mat_deep_copy(a);
    CHECK(mat_equals(copy, a));
    CHECK(copy->rows == 2 && copy->cols == 3);

    mat_free_mat(a);
    mat_free_mat(copy);
    TEST_END();
}

int main(void) {
    printf("mat_copy:\n");

    test_mat_copy_dimensions();
    test_mat_deep_copy_values();
    test_mat_deep_copy_independent();
    test_mat_deep_copy_rectangular();

    TEST_SUMMARY();
}
