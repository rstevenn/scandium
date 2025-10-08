#include "scandium.h"

#include <time.h>
#define STRESS_TEST_ITERATIONS 100



void stress_test(sc_vector* a, sc_vector* b) {
    ccb_arena* tmp_arena = ccb_init_arena();
    sc_vector* out;
    sc_value_t scalar;

    for (int i = 0; i < STRESS_TEST_ITERATIONS; i++) {
        if (i % 10 == 0) printf("Iteration %d/%d\n", i, STRESS_TEST_ITERATIONS);
        out = sc_vector_add(a, b, tmp_arena);
        out = sc_vector_add_inplace(a, b); 
        out = sc_vector_sub(a, b, tmp_arena);
        out = sc_vector_sub_inplace(a, b);
        out = sc_vector_mul_ellement_wise(a, b, tmp_arena);
        out = sc_vector_mul_ellement_wise_inplace(a, b);
        out = sc_vector_div_ellement_wise(a, b, tmp_arena);
        out = sc_vector_div_ellement_wise_inplace(a, b);
        scalar = sc_vector_dot(a, b);

        out = sc_vector_normalize(a, tmp_arena);
        out = sc_vector_normalize_inplace(a, tmp_arena);

        out = sc_vector_map(a, sc_scalar_abs, tmp_arena);
        out = sc_vector_map_inplace(a, sc_scalar_abs);
        out = sc_vector_map_args(a, sc_scalar_add_args, tmp_arena, &scalar);
        out = sc_vector_map_args_inplace(a, sc_scalar_add_args, &scalar);

        ccb_arena_reset(tmp_arena);
    }
    
    ccb_arena_free(tmp_arena);
}


int main(int argc, char** argv) {
    ccb_InitLog("log/perfs.log");
    CCB_INFO("suports avx %d", __builtin_cpu_supports("avx"))
    CCB_INFO("suports avx2 %d", __builtin_cpu_supports("avx2"))




    ccb_arena* arena = ccb_init_arena();
    CCB_NOTNULL(arena, "Failed to create arena");

    // Example usage of scandium library
    uint64_t size = 10000000;
    uint64_t op_count = 19*size;

    sc_vector* vec1 = sc_create_vector(size, sc_float32, arena);
    sc_vector* vec2 = sc_create_vector(size, sc_float32, arena);
    sc_vector* result = sc_create_vector(size, sc_float32, arena);
    CCB_NOTNULL(vec1, "Failed to create vector 1");
    CCB_NOTNULL(vec2, "Failed to create vector 2");

    // Initialize vectors
    float* data1 = (float*)vec1->data;
    float* data2 = (float*)vec2->data;
    for (uint64_t i = 0; i < size; i++) {
        data1[i] = (float)i;
        data2[i] = (float)(size - i);
    }

    // Perform element-wise addition
    clock_t start = clock();
    stress_test(vec1, vec2);
    clock_t end = clock();

    double time_spent = ((double)end - (double)start)/CLOCKS_PER_SEC / STRESS_TEST_ITERATIONS;
    printf("Stress test took %f seconds per iteration.\n", time_spent);
    
    char letters[] = "kMGTP";
    float reminder =  op_count/time_spent;
    char letter = ' ';
    for (int i = 0; i < 5; i++) {
        if (reminder < 1000) {
            break;
        }
        letter = letters[i];
        reminder /= 1000;
    }

    printf("Engine speed: %.02f %cop/s\n", reminder, letter);

    // Clean up
    ccb_arena_free(arena);
    return 0;
}