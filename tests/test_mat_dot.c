#define MATDEF static inline
#define MAT_IMPLEMENTATION
#include "mat.h"
#include "test.h"

void test_mat_dot_basic(void) {
    TEST_BEGIN("mat_dot basic");
    Vec *a = mat_vec_from(3, (mat_elem_t[]){1, 2, 3});
    Vec *b = mat_vec_from(3, (mat_elem_t[]){4, 5, 6});
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    mat_elem_t result = mat_dot(a, b);
    CHECK_FLOAT_EQ(result, 32);

    mat_free_mat(a);
    mat_free_mat(b);
    TEST_END();
}

void test_mat_dot_orthogonal(void) {
    TEST_BEGIN("mat_dot orthogonal vectors");
    Vec *a = mat_vec_from(2, (mat_elem_t[]){1, 0});
    Vec *b = mat_vec_from(2, (mat_elem_t[]){0, 1});
    // Orthogonal vectors have dot product 0
    mat_elem_t result = mat_dot(a, b);
    CHECK_FLOAT_EQ(result, 0);

    mat_free_mat(a);
    mat_free_mat(b);
    TEST_END();
}

void test_mat_dot_self(void) {
    TEST_BEGIN("mat_dot self equals magnitude squared");
    Vec *a = mat_vec_from(3, (mat_elem_t[]){3, 4, 0});
    // 3*3 + 4*4 + 0*0 = 9 + 16 = 25
    mat_elem_t result = mat_dot(a, a);
    CHECK_FLOAT_EQ(result, 25);

    mat_free_mat(a);
    TEST_END();
}

void test_mat_dot_commutative(void) {
    TEST_BEGIN("mat_dot commutative");
    Vec *a = mat_vec_from(3, (mat_elem_t[]){1, 2, 3});
    Vec *b = mat_vec_from(3, (mat_elem_t[]){4, 5, 6});

    mat_elem_t ab = mat_dot(a, b);
    mat_elem_t ba = mat_dot(b, a);
    CHECK_FLOAT_EQ(ab, ba);

    mat_free_mat(a);
    mat_free_mat(b);
    TEST_END();
}

void test_mat_dot_zero(void) {
    TEST_BEGIN("mat_dot with zero vector");
    Vec *a = mat_vec_from(3, (mat_elem_t[]){1, 2, 3});
    Vec *z = mat_vec(3);

    mat_elem_t result = mat_dot(a, z);
    CHECK_FLOAT_EQ(result, 0);

    mat_free_mat(a);
    mat_free_mat(z);
    TEST_END();
}

int main(void) {
    printf("mat_dot:\n");

    test_mat_dot_basic();
    test_mat_dot_orthogonal();
    test_mat_dot_self();
    test_mat_dot_commutative();
    test_mat_dot_zero();

    TEST_SUMMARY();
}
