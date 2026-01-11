#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

void test_mat_diag_extract(void) {
    TEST_BEGIN("mat_diag extract diagonal");
    Mat *a = mat_from(3, 3, (mat_elem_t[]){
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });
    Vec *expected = mat_vec_from(3, (mat_elem_t[]){1, 5, 9});

    Vec *diag = mat_diag(a);
    CHECK(mat_equals(diag, expected));

    mat_free_mat(a);
    mat_free_mat(expected);
    mat_free_mat(diag);
    TEST_END();
}

void test_mat_diag_from_basic(void) {
    TEST_BEGIN("mat_diag_from creates diagonal matrix");
    Mat *expected = mat_from(3, 3, (mat_elem_t[]){
        1, 0, 0,
        0, 2, 0,
        0, 0, 3
    });

    Mat *diag = mat_diag_from(3, (mat_elem_t[]){1, 2, 3});
    CHECK(mat_equals(diag, expected));

    mat_free_mat(expected);
    mat_free_mat(diag);
    TEST_END();
}

void test_mat_diag_from_eye(void) {
    TEST_BEGIN("mat_diag_from with ones equals eye");
    Mat *eye = mat_reye(3);

    Mat *diag = mat_diag_from(3, (mat_elem_t[]){1, 1, 1});
    CHECK(mat_equals(diag, eye));

    mat_free_mat(eye);
    mat_free_mat(diag);
    TEST_END();
}

void test_mat_diag_roundtrip(void) {
    TEST_BEGIN("mat_diag roundtrip");
    mat_elem_t values[] = {1, 2, 3, 4};

    Mat *diag_mat = mat_diag_from(4, values);
    Vec *extracted = mat_diag(diag_mat);
    Vec *expected = mat_vec_from(4, values);

    CHECK(mat_equals(extracted, expected));

    mat_free_mat(diag_mat);
    mat_free_mat(extracted);
    mat_free_mat(expected);
    TEST_END();
}

void test_mat_diag_eye(void) {
    TEST_BEGIN("mat_diag of eye equals ones vector");
    Mat *eye = mat_reye(3);
    Vec *expected = mat_vec_from(3, (mat_elem_t[]){1, 1, 1});

    Vec *diag = mat_diag(eye);
    CHECK(mat_equals(diag, expected));

    mat_free_mat(eye);
    mat_free_mat(expected);
    mat_free_mat(diag);
    TEST_END();
}

int main(void) {
    printf("mat_diag:\n");

    test_mat_diag_extract();
    test_mat_diag_from_basic();
    test_mat_diag_from_eye();
    test_mat_diag_roundtrip();
    test_mat_diag_eye();

    TEST_SUMMARY();
}
