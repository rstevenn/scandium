#include "const.h"
#include "../ccbase/utils/mem.h"
#include "../ccbase/logs/log.h" 
#include <string.h>
#include "data.h"
#include "linalg.h"
#include <math.h>

int test_dims_creation_(ccb_arena* arena){
    sc_dimensions* dims = sc_create_dimensions(2, arena, (uint32_t[]){2, 5});
    if (!dims) {
        CCB_WARNING("Failed to create dimensions");
        return -1;
    }
    for (uint32_t i = 0; i < 2; i++) {
        if (dims->dims[i] != (uint32_t[]){2, 5}[i]) {
            CCB_WARNING("Dimensions creation failed");
            return -1;
        }
    }
    return 0;
}

int test_indices_creation_(ccb_arena* arena){
    sc_index* index = sc_create_index(2, arena, (uint32_t[]){1, 2});
    if (!index) {
        CCB_WARNING("Failed to create index");
        return -1;
    }
    for (uint32_t i = 0; i < 2; i++) {
        if (index->indices[i] != (uint32_t[]){1, 2}[i]) {
            CCB_WARNING("Index creation failed");
            return -1;
        }
    }
    return 0;
}

int test_slice_creation_(ccb_arena* arena){
    sc_slice_t* slice = sc_create_slice(2, arena, (uint32_t[]){1, 2}, (uint32_t[]){3, 4});
    if (!slice) {
        CCB_WARNING("Failed to create slice");
        return -1;
    }
    for (uint32_t i = 0; i < 2; i++) {
        if (slice->slices[i].start != (uint32_t[]){1, 2}[i] || slice->slices[i].end != (uint32_t[]){3, 4}[i]) {
            CCB_WARNING("Slice creation failed");
            return -1;
        }
    }
    return 0;
}

int test_vector_creation___bf16(ccb_arena* arena){
    sc_vector* vector = sc_create_vector(10, sc_float16, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }
    return 0;
}

int test_tensor_creation___bf16(ccb_arena* arena){
    sc_dimensions* dims = sc_create_dimensions(2, arena, (uint32_t[]){2, 5});
    if (!dims) {
        CCB_WARNING("Failed to create dimensions");
        return -1;
    }

    sc_tensor* tensor = sc_create_tensor(dims, sc_float16, arena);
    if (!tensor) {
        CCB_WARNING("Failed to create tensor");
        return -1;
    }
    return 0;
}

int test_vector_data_loading___bf16(ccb_arena* arena){
    sc_vector* vector = sc_create_vector(10, sc_float16, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    __bf16 data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    sc_data_to_vector(vector, data, 10);
    return 0;
}

int test_tensor_data_loading___bf16(ccb_arena* arena){
    sc_dimensions* dims = sc_create_dimensions(2, arena, (uint32_t[]){2, 5});
    if (!dims) {
        CCB_WARNING("Failed to create dimensions");
        return -1;
    }

    sc_tensor* tensor = sc_create_tensor(dims, sc_float16, arena);
    if (!tensor) {
        CCB_WARNING("Failed to create tensor");
        return -1;
    }

    __bf16 data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    sc_data_to_tensor(tensor, data, 10);
    return 0;
}

int test_vector_clone___bf16(ccb_arena* arena) {
    sc_vector* vector = sc_create_vector(10, sc_float16, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((__bf16)i, sc_float16);
        sc_set_vector_element(vector, i, val);
    }

    sc_vector* clone = sc_clone_vector(vector, arena);
    if (!clone) {
        CCB_WARNING("Failed to clone vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t original_val = sc_get_vector_element(vector, i);
        sc_value_t clone_val = sc_get_vector_element(clone, i);
        if (original_val.value.f16 != clone_val.value.f16) {
            CCB_WARNING("Vector clone failed");
            return -1;
        }
    }

    return 0;
}

int test_tensor_clone___bf16(ccb_arena* arena) {
    sc_dimensions* dims = sc_create_dimensions(2, arena, (uint32_t[]){3, 4});
    if (!dims) {
        CCB_WARNING("Failed to create dimensions");
        return -1;
    }

    sc_tensor* tensor = sc_create_tensor(dims, sc_float16, arena);
    if (!tensor) {
        CCB_WARNING("Failed to create tensor");
        return -1;
    }

    for (uint32_t i = 0; i < 3; i++) {
        for (uint32_t j = 0; j < 4; j++) {
            sc_index index;
            index.count = 2;
            uint32_t idxs[2] = {i, j};
            index.indices = idxs;
            sc_value_t val = to_sc_value((__bf16)(i * 4 + j), sc_float16);
            sc_set_tensor_element(tensor, &index, val);
        }
    }

    sc_tensor* clone = sc_clone_tensor(tensor, arena);
    if (!clone) {
        CCB_WARNING("Failed to clone tensor");
        return -1;
    }

    for (uint32_t i = 0; i < 3; i++) {
        for (uint32_t j = 0; j < 4; j++) {
            sc_index index;
            index.count = 2;
            uint32_t idxs[2] = {i, j};
            index.indices = idxs;

            sc_value_t original_val = sc_get_tensor_element(tensor, &index);
            sc_value_t clone_val = sc_get_tensor_element(clone, &index);
            if (original_val.value.f16 != clone_val.value.f16) {
                CCB_WARNING("tensor clone failed");
                return -1;
            }
        }
    }

    return 0;
}

int test_vector_set_get___bf16(ccb_arena* arena) {
    sc_vector* vector = sc_create_vector(10, sc_float16, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((__bf16)i, sc_float16);
        sc_set_vector_element(vector, i, val);
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(vector, i);
        if (val.type != sc_float16 || val.value.f16 != (__bf16)i) {
            CCB_WARNING("Vector set/get mismatch at index %u: expected %f, got %f", i, (__bf16)i, val.value.f16);
            return -1;
        }
    }

    return 0;
}

int test_tensor_set_get___bf16(ccb_arena* arena) {
    sc_dimensions* dims = sc_create_dimensions(2, arena, (uint32_t[]){2, 5});
    if (!dims) {
        CCB_WARNING("Failed to create dimensions");
        return -1;
    }

    sc_tensor* tensor = sc_create_tensor(dims, sc_float16, arena);
    if (!tensor) {
        CCB_WARNING("Failed to create tensor");
        return -1;
    }

    for (uint32_t i = 0; i < 2; i++) {
        for (uint32_t j = 0; j < 5; j++) {
            sc_index index;
            index.count = 2;
            uint32_t idxs[2] = {i, j};
            index.indices = idxs;
            sc_value_t val = to_sc_value((__bf16)(i * 5 + j) * 3.0, sc_float16);
            sc_set_tensor_element(tensor, &index, val);
        }
    }

    for (uint32_t i = 0; i < 2; i++) {
        for (uint32_t j = 0; j < 5; j++) {
            sc_index index;
            index.count = 2;
            uint32_t idxs[2] = {i, j};
            index.indices = idxs;

            sc_value_t val = sc_get_tensor_element(tensor, &index);
            if (val.type != sc_float16 || val.value.f16 != ((__bf16)(i * 5 + j) * 3.0)) {
                CCB_WARNING("tensor set/get mismatch at index [%u, %u]: expected %f, got %f", i, j, (__bf16)(i * 5 + j) * 3.0, val.value.f16);
                return -1;
            }
        }
    }

    return 0;
}

int test_vector_add___bf16(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float16, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float16, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector2");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((__bf16)i, sc_float16);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((__bf16)(i * 2), sc_float16);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_vector* result = sc_vector_add(vector1, vector2, arena);
    if (!result) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        if (val.type != sc_float16 || val.value.f16 != (__bf16)(i + i * 2)) {
            CCB_WARNING("Vector addition mismatch at index %u: expected %f, got %f", i, (__bf16)(i + i * 2), val.value.f16);
            return -1;
        }
    }

    return 0;
}

int test_vector_add_inplace___bf16(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float16, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float16, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector2");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((__bf16)i, sc_float16);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((__bf16)(i * 2), sc_float16);
        sc_set_vector_element(vector2, i, val2);
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = sc_get_vector_element(vector1, i);
        sc_value_t val2 = sc_get_vector_element(vector2, i);
        sc_value_t sum;
        sum.type = sc_float16;
        sum.value.f16 = (__bf16)((float)val1.value.f16 + (float)val2.value.f16);
        sc_set_vector_element(vector1, i, sum);
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(vector1, i);
        if (val.type != sc_float16 || val.value.f16 != (__bf16)(i + i * 2)) {
            CCB_WARNING("In-place vector addition mismatch at index %u : expected %f, got %f", i, (__bf16)(i + i * 2), val.value.f16);
            return -1;
        }
    }

    return 0;
}

int test_vector_add_scalar___bf16(ccb_arena* arena) {
    sc_vector* vector = sc_create_vector(10, sc_float16, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((__bf16)i, sc_float16);
        sc_set_vector_element(vector, i, val);
    }

    sc_value_t scalar = to_sc_value((__bf16)5, sc_float16);
    sc_vector* result = sc_vector_add_scalar(vector, scalar, arena);
    if (!result) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        if (val.type != sc_float16 || val.value.f16 != (__bf16)(i + 5)) {
            CCB_WARNING("Vector-scalar addition mismatch at index %u   : expected %f, got %f", i, (__bf16)(i + 5), val.value.f16);
            return -1;
        }
    }

    return 0;
}

int test_vector_add_scalar_inplace___bf16(ccb_arena* arena) {
    sc_vector* vector = sc_create_vector(10, sc_float16, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((__bf16)i, sc_float16);
        sc_set_vector_element(vector, i, val);
    }

    sc_value_t scalar = to_sc_value((__bf16)5, sc_float16);
    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(vector, i);
        sc_value_t sum;
        sum.type = sc_float16;
        sum.value.f16 = val.value.f16 + scalar.value.f16;
        sc_set_vector_element(vector, i, sum);
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(vector, i);
        if (val.type != sc_float16 || val.value.f16 != (__bf16)(i + 5)) {
            CCB_WARNING("In-place vector-scalar addition mismatch at index %u : expected %f, got %f", i, (__bf16)(i + 5), val.value.f16);
            return -1;
        }
    }

    return 0;
}

int test_vector_sub___bf16(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float16, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float16, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector2");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((__bf16)(i * 3), sc_float16);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((__bf16)(i * 2), sc_float16);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_vector* result = sc_vector_sub(vector1, vector2, arena);
    if (!result) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        if (val.type != sc_float16 || val.value.f16 != (__bf16)(i * 3 - i * 2)) {
            CCB_WARNING("Vector subtraction mismatch at index %u: expected %f, got %f", i, (__bf16)(i * 3 - i * 2), val.value.f16);
            return -1;
        }
    }

    return 0;
}

int test_vector_sub_inplace___bf16(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float16, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float16, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector2");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((__bf16)(i * 3), sc_float16);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((__bf16)(i * 2), sc_float16);
        sc_set_vector_element(vector2, i, val2);
    }

    vector1 = sc_vector_sub_inplace(vector1, vector2);
    if (!vector1) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(vector1, i);
        if (val.type != sc_float16 || val.value.f16 != (__bf16)(i * 3 - i * 2)) {
            CCB_WARNING("Vector subtraction mismatch at index %u: expected %f, got %f", i, (__bf16)(i * 3 - i * 2), val.value.f16);
            return -1;
        }
    }

    return 0;
}

int test_vector_sub_scalar___bf16(ccb_arena* arena) {
    sc_vector* vector = sc_create_vector(10, sc_float16, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((__bf16)(i * 3), sc_float16);
        sc_set_vector_element(vector, i, val);
    }

    sc_value_t scalar = to_sc_value((__bf16)2, sc_float16);
    sc_vector* result = sc_vector_sub_scalar(vector, scalar, arena);
    if (!result) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (int32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        if (val.type != sc_float16 || val.value.f16 != (__bf16)(i * 3 - 2)) {
            CCB_WARNING("Vector-scalar subtraction mismatch at index %u: expected %f, got %f", i, (__bf16)(i * 3 - 2), val.value.f16);
            return -1;
        }
    }

    return 0;
}

int test_vector_sub_scalar_inplace___bf16(ccb_arena* arena) {
    sc_vector* vector = sc_create_vector(10, sc_float16, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((__bf16)(i * 3), sc_float16);
        sc_set_vector_element(vector, i, val);
    }

    sc_value_t scalar = to_sc_value((__bf16)2, sc_float16);
    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(vector, i);
        val.value.f16 -= scalar.value.f16;
        sc_set_vector_element(vector, i, val);
    }

    for (int32_t i = 0; i < 10; i++) {
         sc_value_t val = sc_get_vector_element(vector, i);
         if (val.type != sc_float16 || val.value.f16 != (__bf16)(i * 3 - 2)) {
             CCB_WARNING("Vector-scalar subtraction mismatch at index %u: expected %f, got %f", i, (__bf16)(i * 3 - 2), val.value.f16);
             return -1;
         }
    }

    return 0;
}

int test_vector_mul_ellement_wise___bf16(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float16, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float16, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector2");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((__bf16)i, sc_float16);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((__bf16)(i * 2), sc_float16);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_vector* result = sc_vector_mul_ellement_wise(vector1, vector2, arena);
    if (!result) {
        CCB_WARNING("Failed to create result vectore");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        if (val.type != sc_float16 || val.value.f16 != (__bf16)(i * i * 2)) {
            CCB_WARNING("Vector element-wise multiplication mismatch at index %u: expected %f, got %f", i, (__bf16)(i * i * 2), val.value.f16);
            return -1;
        }
    }

    return 0;
}

int test_vector_mul_ellement_wise_inplace___bf16(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float16, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float16, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector2");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((__bf16)i, sc_float16);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((__bf16)(i * 2), sc_float16);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_vector* result = sc_vector_mul_ellement_wise_inplace(vector1, vector2);
    if (!result) {
        CCB_WARNING("Failed to create result vectore");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        if (val.type != sc_float16 || val.value.f16 != (__bf16)(i * i * 2)) {
            CCB_WARNING("Vector element-wise multiplication mismatch at index %u: expected %f, got %f", i, (__bf16)(i * i * 2), val.value.f16);
            return -1;
        }
    }

    return 0;
}

int test_vector_mul_scalar___bf16(ccb_arena* arena) {
    sc_vector* vector = sc_create_vector(10, sc_float16, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((__bf16)i, sc_float16);
        sc_set_vector_element(vector, i, val);
    }

    sc_value_t scalar = to_sc_value((__bf16)2, sc_float16);
    sc_vector* result = sc_vector_mul_scalar(vector, scalar, arena);
    if (!result){
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        if (val.type != sc_float16 || val.value.f16 != (__bf16)(2 * i)) {
            CCB_WARNING("Vector-scalar multiplication mismatch at index %u: expected %f, got %f", i, (__bf16)(2 * i), val.value.f16);
            return -1;
        }
    }

    return 0;
}

int test_vector_mul_scalar_inplace___bf16(ccb_arena* arena) {
    sc_vector* vector = sc_create_vector(10, sc_float16, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((__bf16)i, sc_float16);
        sc_set_vector_element(vector, i, val);
    }

    sc_value_t scalar = to_sc_value((__bf16)2, sc_float16);
    sc_vector* result = sc_vector_mul_scalar_inplace(vector, scalar);
    if (!result){
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        if (val.type != sc_float16 || val.value.f16 != (__bf16)(2 * i)) {
            CCB_WARNING("Vector-scalar multiplication mismatch at index %u: expected %f, got %f", i, (__bf16)(2 * i), val.value.f16);
            return -1;
        }
    }

    return 0;
}

int test_vector_div_element_wise___bf16(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float16, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float16, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector2");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((__bf16)i, sc_float16);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((__bf16)(i+1), sc_float16);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_vector* result = sc_vector_div_ellement_wise(vector1, vector2, arena);
    if (!result) {
        CCB_WARNING("Failed to create result vectore");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        __bf16 expected = (__bf16)i / (__bf16)(i +1);
        if (val.type != sc_float16 || val.value.f16 != expected) {
            CCB_WARNING("Vector element-wise multiplication mismatch at index %u: expected %f, got %f", i, expected, (float)val.value.f16);
            return -1;
        }
    }

    return 0;
}

int test_vector_div_element_wise_inplace___bf16(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float16, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float16, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector2");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((__bf16)i, sc_float16);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((__bf16)(i+1), sc_float16);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_vector* result = sc_vector_div_ellement_wise_inplace(vector1, vector2);
    if (!result) {
        CCB_WARNING("Failed to create result vectore");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        __bf16 expected = (__bf16)i / (__bf16)(i +1);
        if (val.type != sc_float16 || val.value.f16 != expected) {
            CCB_WARNING("Vector element-wise multiplication mismatch at index %u: expected %f, got %f", i, expected, (float)val.value.f16);
            return -1;
        }
    }

    return 0;
}

int test_vector_div_scalar___bf16(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float16, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((__bf16)i, sc_float16);
        sc_set_vector_element(vector1, i, val);
    }

    sc_value_t scalar = to_sc_value((__bf16)2, sc_float16);
    sc_vector* result = sc_vector_div_scalar(vector1, scalar, arena);
    if (!result) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        __bf16 expected = (__bf16)i / (__bf16)2;
        if (val.type != sc_float16 || val.value.f16 != expected) {
            CCB_WARNING("Vector-scalar division mismatch at index %u: expected %f, got %f", i, expected, (float)val.value.f16);
            return -1;
        }
    }

    return 0;
}

int test_vector_div_scalar_inplace___bf16(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float16, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((__bf16)i, sc_float16);
        sc_set_vector_element(vector1, i, val);
    }

    sc_value_t scalar = to_sc_value((__bf16)2, sc_float16);
    sc_vector* result = sc_vector_div_scalar_inplace(vector1, scalar);
    if (!result) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        __bf16 expected = (__bf16)i / (__bf16)2;
        if (val.type != sc_float16 || val.value.f16 != expected) {
            CCB_WARNING("Vector-scalar division mismatch at index %u: expected %f, got %f", i, expected, (float)val.value.f16);
            return -1;
        }
    }

    return 0;
}

int test_vector_dot_product___bf16(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float16, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float16, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((__bf16)i/2, sc_float16);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((__bf16)(i * 2), sc_float16);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_value_t result = sc_vector_dot(vector1, vector2);

    __bf16 expected = 0;
    for (uint32_t i = 0; i < 10; i++) {
        expected += (__bf16)i/2 * (__bf16)(i * 2);
    }

    if (result.type != sc_float16 || result.value.f16 != expected) {
        CCB_WARNING("Vector dot product mismatch: expected %f, got %f", (float)expected, (float)result.value.f16);
        return -1;
    }

    return 0;
}

int test_vector_cross_product___bf16(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(3, sc_float16, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(3, sc_float16, arena);
    if (!vector2){
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    for (uint32_t i = 0; i < 3; i++) {
        sc_value_t val1 = to_sc_value((__bf16)i, sc_float16);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((__bf16)(i + 1), sc_float16);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_vector* result = sc_vector_cross(vector1, vector2, arena);
    if (!result) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    sc_vector* expected = sc_create_vector(3, sc_float16, arena);
    if (!expected) {
        CCB_WARNING("Failed to create expected vector");
        return -1;
    }

    __bf16* a = (__bf16*)vector1->data;
    __bf16* b = (__bf16*)vector2->data;
    __bf16* c = (__bf16*)expected->data;
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];

    for (uint32_t i = 0; i < 3; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        sc_value_t expected_val = sc_get_vector_element(expected, i);
        if (val.type != sc_float16 || val.value.f16 != expected_val.value.f16) {
            CCB_WARNING("Vector dot product mismatch: expected %f, got %f", (float)expected_val.value.f16, (float)val.value.f16);
            return -1;
         }
    }

    return 0;
}

int test_vector_cross_product_inplace___bf16(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(3, sc_float16, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(3, sc_float16, arena);
    if (!vector2){
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    for (uint32_t i = 0; i < 3; i++) {
        sc_value_t val1 = to_sc_value((__bf16)i, sc_float16);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((__bf16)(i + 1), sc_float16);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_vector* expected = sc_create_vector(3, sc_float16, arena);
    if (!expected) {
        CCB_WARNING("Failed to create expected vector");
        return -1;
    }

    __bf16* a = (__bf16*)vector1->data;
    __bf16* b = (__bf16*)vector2->data;
    __bf16* c = (__bf16*)expected->data;
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];

    sc_vector* result = sc_vector_cross_inplace(vector1, vector2);
    if (!result) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 3; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        sc_value_t expected_val = sc_get_vector_element(expected, i);
        if (val.type != sc_float16 || val.value.f16 != expected_val.value.f16) {
            CCB_WARNING("Vector dot product mismatch: expected %f, got %f", (float)expected_val.value.f16, (float)val.value.f16);
            return -1;
         }
    }

    return 0;
}

int test_vector_norm1___bf16(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float16, arena);
    CCB_NOTNULL(vector1, "Failed to create vector1");

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((__bf16)i, sc_float16);
        sc_set_vector_element(vector1, i, val);
    }

    sc_value_t result = sc_vector_norm(vector1, 1, arena);
    if (result.type != sc_float16) {
        CCB_WARNING("Vector norm mismatch: expected %f, got %f", (float)0, (float)result.value.f16);
        return -1;
    }

    sc_value_t expected = to_sc_value((__bf16)0, sc_float16);
    for (uint32_t i = 0; i < 10; i++) {
        expected = sc_scalar_add(expected, sc_scalar_abs(sc_get_vector_element(vector1, i)));
    }

    if (result.value.f16 != expected.value.f16) {
         CCB_WARNING("Vector norm mismatch: expected %f, got %f", (float)expected.value.f16, (float)result.value.f16);
         return -1;
    }

    return 0;
}

int test_vector_norm2___bf16(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float16, arena);
    CCB_NOTNULL(vector1, "Failed to create vector1");

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((__bf16)i, sc_float16);
        sc_set_vector_element(vector1, i, val);
    }

    sc_value_t result = sc_vector_norm(vector1, 2, arena);
    if (result.type != sc_float16) {
        CCB_WARNING("Vector norm mismatch: expected %f, got %f", (float)0, (float)result.value.f16);
        return -1;
    }

    sc_value_t expected = to_sc_value((__bf16)0, sc_float16);
    sc_value_t val_p = to_sc_value((__bf16)2, sc_float16);
    for (uint32_t i = 0; i < 10; i++) {
        expected = sc_scalar_add(expected, sc_scalar_pow(sc_get_vector_element(vector1, i), val_p));
    }

    expected = sc_scalar_root(expected, val_p);
    if (result.value.f16 != expected.value.f16) {
         CCB_WARNING("Vector norm mismatch: expected %f, got %f", (float)expected.value.f16, (float)result.value.f16);
         return -1;
    }

    return 0;
}

int test_vector_norm3___bf16(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float16, arena);
    CCB_NOTNULL(vector1, "Failed to create vector1");

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((__bf16)i, sc_float16);
        sc_set_vector_element(vector1, i, val);
    }

    sc_value_t result = sc_vector_norm(vector1, 3, arena);
    if (result.type != sc_float16) {
        CCB_WARNING("Vector norm mismatch: expected %f, got %f", (float)0, (float)result.value.f16);
        return -1;
    }

    sc_value_t expected = to_sc_value((__bf16)0, sc_float16);
    sc_value_t val_p = to_sc_value((__bf16)3, sc_float16);
    for (uint32_t i = 0; i < 10; i++) {
        expected = sc_scalar_add(expected, sc_scalar_pow(sc_get_vector_element(vector1, i), val_p));
    }

    expected = sc_scalar_root(expected, val_p);
    if (result.value.f16 != expected.value.f16) {
         CCB_WARNING("Vector norm mismatch: expected %f, got %f", (float)expected.value.f16, (float)result.value.f16);
         return -1;
    }

    return 0;
}

int test_vector_normalization___bf16(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float16, arena);
    CCB_NOTNULL(vector1, "Failed to create vector1");

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((__bf16)i, sc_float16);
        sc_set_vector_element(vector1, i, val);
    }

    sc_vector* result = sc_vector_normalize(vector1, arena);
    if (!result) {
         CCB_WARNING("Failed to create result vector");
         return -1;
    }

    sc_value_t norm = sc_vector_norm(vector1, 2, arena);
    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        sc_value_t expected = sc_scalar_div(sc_get_vector_element(vector1, i), norm);
        if (val.type != sc_float16 || val.value.f16 != expected.value.f16) {
            CCB_WARNING("Vector normalization mismatch at index %u: expected %f, got %f", i, (float)expected.value.f16, (float)val.value.f16);
            return -1;
        }
    }

    return 0;
}

int test_vector_normalization_inplace___bf16(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float16, arena);
    CCB_NOTNULL(vector1, "Failed to create vector1");

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((__bf16)i, sc_float16);
        sc_set_vector_element(vector1, i, val);
    }

    sc_value_t norm = sc_vector_norm(vector1, 2, arena);
    sc_vector* result = sc_vector_normalize_inplace(vector1, arena);
    if (!result) {
         CCB_WARNING("Failed to create result vector");
         return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        sc_value_t expected = sc_scalar_div(to_sc_value((__bf16)i, sc_float16), norm);
        if (val.type != sc_float16 || val.value.f16 != expected.value.f16) {
            CCB_WARNING("Vector normalization mismatch at index %u: expected %f, got %f", i, (float)expected.value.f16, (float)val.value.f16);
            return -1;
        }
    }

    return 0;
}

int test_get_sub_tensor___bf16(ccb_arena* arena) {
    uint32_t shape[3] = {4, 4, 4};
    sc_dimensions* dims = sc_create_dimensions(3, arena, shape);
    sc_tensor* tensor = sc_create_tensor(dims, sc_float16, arena);
    CCB_NOTNULL(tensor, "Failed to create tensor");

    sc_index* zero_idx = sc_create_index(3, arena, (uint32_t[]){0, 0, 0});
    CCB_NOTNULL(zero_idx, "Failed to create zero index");

    for (uint32_t i = 0; i < 64; i++) {
        sc_value_t val = to_sc_value((__bf16)i, sc_float16);
        for (uint32_t j = 0; j < 3; j++) {
            zero_idx->indices[j] = (i / (uint32_t)pow(4, 2 - j)) % 4;
        }
        sc_set_tensor_element(tensor, zero_idx, val);
    }

    uint32_t start[3] = {1, 1};
    sc_index* indices = sc_create_index(2, arena, start);
    CCB_NOTNULL(indices, "Failed to create indices");

    sc_tensor* sub_tensor = sc_get_sub_tensor(tensor, indices, arena);
    CCB_NOTNULL(sub_tensor, "Failed to create sub-tensor");

    CCB_INFO("Sub-tensor allocated: %p", sub_tensor);
    uint32_t expected_shape[1] = {4};
    if (sub_tensor->dims->dims_count != 1 || memcmp(sub_tensor->dims->dims, expected_shape, sizeof(expected_shape)) != 0) {
        CCB_WARNING("Sub-tensor shape mismatch: expected [4], got [%u, %u, %u]", sub_tensor->dims->dims[0], sub_tensor->dims->dims[1], sub_tensor->dims->dims[2]);
        return -1;
    }

    for (uint32_t i = 0; i < 4; i++) {
        sc_index* idx = sc_create_index(1, arena, &i);
        sc_value_t val = sc_get_tensor_element(sub_tensor, idx);
        __bf16 expected = (__bf16)(16 + i);
        if (val.type != sc_float16 || val.value.f16 != expected) {
            CCB_WARNING("Sub-tensor element mismatch at index %u: expected %f, got %f", i, (float)expected, (float)val.value.f16);
            return -1;
        }
    }

    return 0;
}

int test_vector_creation_float(ccb_arena* arena){
    sc_vector* vector = sc_create_vector(10, sc_float32, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }
    return 0;
}

int test_tensor_creation_float(ccb_arena* arena){
    sc_dimensions* dims = sc_create_dimensions(2, arena, (uint32_t[]){2, 5});
    if (!dims) {
        CCB_WARNING("Failed to create dimensions");
        return -1;
    }

    sc_tensor* tensor = sc_create_tensor(dims, sc_float32, arena);
    if (!tensor) {
        CCB_WARNING("Failed to create tensor");
        return -1;
    }
    return 0;
}

int test_vector_data_loading_float(ccb_arena* arena){
    sc_vector* vector = sc_create_vector(10, sc_float32, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    float data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    sc_data_to_vector(vector, data, 10);
    return 0;
}

int test_tensor_data_loading_float(ccb_arena* arena){
    sc_dimensions* dims = sc_create_dimensions(2, arena, (uint32_t[]){2, 5});
    if (!dims) {
        CCB_WARNING("Failed to create dimensions");
        return -1;
    }

    sc_tensor* tensor = sc_create_tensor(dims, sc_float32, arena);
    if (!tensor) {
        CCB_WARNING("Failed to create tensor");
        return -1;
    }

    float data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    sc_data_to_tensor(tensor, data, 10);
    return 0;
}

int test_vector_clone_float(ccb_arena* arena) {
    sc_vector* vector = sc_create_vector(10, sc_float32, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((float)i, sc_float32);
        sc_set_vector_element(vector, i, val);
    }

    sc_vector* clone = sc_clone_vector(vector, arena);
    if (!clone) {
        CCB_WARNING("Failed to clone vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t original_val = sc_get_vector_element(vector, i);
        sc_value_t clone_val = sc_get_vector_element(clone, i);
        if (original_val.value.f32 != clone_val.value.f32) {
            CCB_WARNING("Vector clone failed");
            return -1;
        }
    }

    return 0;
}

int test_tensor_clone_float(ccb_arena* arena) {
    sc_dimensions* dims = sc_create_dimensions(2, arena, (uint32_t[]){3, 4});
    if (!dims) {
        CCB_WARNING("Failed to create dimensions");
        return -1;
    }

    sc_tensor* tensor = sc_create_tensor(dims, sc_float32, arena);
    if (!tensor) {
        CCB_WARNING("Failed to create tensor");
        return -1;
    }

    for (uint32_t i = 0; i < 3; i++) {
        for (uint32_t j = 0; j < 4; j++) {
            sc_index index;
            index.count = 2;
            uint32_t idxs[2] = {i, j};
            index.indices = idxs;
            sc_value_t val = to_sc_value((float)(i * 4 + j), sc_float32);
            sc_set_tensor_element(tensor, &index, val);
        }
    }

    sc_tensor* clone = sc_clone_tensor(tensor, arena);
    if (!clone) {
        CCB_WARNING("Failed to clone tensor");
        return -1;
    }

    for (uint32_t i = 0; i < 3; i++) {
        for (uint32_t j = 0; j < 4; j++) {
            sc_index index;
            index.count = 2;
            uint32_t idxs[2] = {i, j};
            index.indices = idxs;

            sc_value_t original_val = sc_get_tensor_element(tensor, &index);
            sc_value_t clone_val = sc_get_tensor_element(clone, &index);
            if (original_val.value.f32 != clone_val.value.f32) {
                CCB_WARNING("tensor clone failed");
                return -1;
            }
        }
    }

    return 0;
}

int test_vector_set_get_float(ccb_arena* arena) {
    sc_vector* vector = sc_create_vector(10, sc_float32, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((float)i, sc_float32);
        sc_set_vector_element(vector, i, val);
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(vector, i);
        if (val.type != sc_float32 || val.value.f32 != (float)i) {
            CCB_WARNING("Vector set/get mismatch at index %u: expected %f, got %f", i, (float)i, val.value.f32);
            return -1;
        }
    }

    return 0;
}

int test_tensor_set_get_float(ccb_arena* arena) {
    sc_dimensions* dims = sc_create_dimensions(2, arena, (uint32_t[]){2, 5});
    if (!dims) {
        CCB_WARNING("Failed to create dimensions");
        return -1;
    }

    sc_tensor* tensor = sc_create_tensor(dims, sc_float32, arena);
    if (!tensor) {
        CCB_WARNING("Failed to create tensor");
        return -1;
    }

    for (uint32_t i = 0; i < 2; i++) {
        for (uint32_t j = 0; j < 5; j++) {
            sc_index index;
            index.count = 2;
            uint32_t idxs[2] = {i, j};
            index.indices = idxs;
            sc_value_t val = to_sc_value((float)(i * 5 + j) * 3.0f, sc_float32);
            sc_set_tensor_element(tensor, &index, val);
        }
    }

    for (uint32_t i = 0; i < 2; i++) {
        for (uint32_t j = 0; j < 5; j++) {
            sc_index index;
            index.count = 2;
            uint32_t idxs[2] = {i, j};
            index.indices = idxs;

            sc_value_t val = sc_get_tensor_element(tensor, &index);
            if (val.type != sc_float32 || val.value.f32 != ((float)(i * 5 + j) * 3.0f)) {
                CCB_WARNING("tensor set/get mismatch at index [%u, %u]: expected %f, got %f", i, j, (float)(i * 5 + j) * 3.0f, val.value.f32);
                return -1;
            }
        }
    }

    return 0;
}

int test_vector_add_float(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float32, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float32, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector2");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((float)i, sc_float32);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((float)(i * 2), sc_float32);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_vector* result = sc_vector_add(vector1, vector2, arena);
    if (!result) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        if (val.type != sc_float32 || val.value.f32 != (float)(i + i * 2)) {
            CCB_WARNING("Vector addition mismatch at index %u: expected %f, got %f", i, (float)(i + i * 2), val.value.f32);
            return -1;
        }
    }

    return 0;
}

int test_vector_add_inplace_float(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float32, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float32, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector2");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((float)i, sc_float32);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((float)(i * 2), sc_float32);
        sc_set_vector_element(vector2, i, val2);
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = sc_get_vector_element(vector1, i);
        sc_value_t val2 = sc_get_vector_element(vector2, i);
        sc_value_t sum;
        sum.type = sc_float32;
        sum.value.f32 = val1.value.f32 + val2.value.f32;
        sc_set_vector_element(vector1, i, sum);
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(vector1, i);
        if (val.type != sc_float32 || val.value.f32 != (float)(i + i * 2)) {
            CCB_WARNING("In-place vector addition mismatch at index %u : expected %f, got %f", i, (float)(i + i * 2), val.value.f32);
            return -1;
        }
    }

    return 0;
}

int test_vector_add_scalar_float(ccb_arena* arena) {
    sc_vector* vector = sc_create_vector(10, sc_float32, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((float)i, sc_float32);
        sc_set_vector_element(vector, i, val);
    }

    sc_value_t scalar = to_sc_value((float)5, sc_float32);
    sc_vector* result = sc_vector_add_scalar(vector, scalar, arena);
    if (!result) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        if (val.type != sc_float32 || val.value.f32 != (float)(i + 5)) {
            CCB_WARNING("Vector-scalar addition mismatch at index %u   : expected %f, got %f", i, (float)(i + 5), val.value.f32);
            return -1;
        }
    }

    return 0;
}

int test_vector_add_scalar_inplace_float(ccb_arena* arena) {
    sc_vector* vector = sc_create_vector(10, sc_float32, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((float)i, sc_float32);
        sc_set_vector_element(vector, i, val);
    }

    sc_value_t scalar = to_sc_value((float)5, sc_float32);
    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(vector, i);
        sc_value_t sum;
        sum.type = sc_float32;
        sum.value.f32 = val.value.f32 + scalar.value.f32;
        sc_set_vector_element(vector, i, sum);
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(vector, i);
        if (val.type != sc_float32 || val.value.f32 != (float)(i + 5)) {
            CCB_WARNING("In-place vector-scalar addition mismatch at index %u : expected %f, got %f", i, (float)(i + 5), val.value.f32);
            return -1;
        }
    }

    return 0;
}

int test_vector_sub_float(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float32, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float32, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector2");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((float)(i * 3), sc_float32);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((float)(i * 2), sc_float32);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_vector* result = sc_vector_sub(vector1, vector2, arena);
    if (!result) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        if (val.type != sc_float32 || val.value.f32 != (float)(i * 3 - i * 2)) {
            CCB_WARNING("Vector subtraction mismatch at index %u: expected %f, got %f", i, (float)(i * 3 - i * 2), val.value.f32);
            return -1;
        }
    }

    return 0;
}

int test_vector_sub_inplace_float(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float32, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float32, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector2");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((float)(i * 3), sc_float32);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((float)(i * 2), sc_float32);
        sc_set_vector_element(vector2, i, val2);
    }

    vector1 = sc_vector_sub_inplace(vector1, vector2);
    if (!vector1) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(vector1, i);
        if (val.type != sc_float32 || val.value.f32 != (float)(i * 3 - i * 2)) {
            CCB_WARNING("Vector subtraction mismatch at index %u: expected %f, got %f", i, (float)(i * 3 - i * 2), val.value.f32);
            return -1;
        }
    }

    return 0;
}

int test_vector_sub_scalar_float(ccb_arena* arena) {
    sc_vector* vector = sc_create_vector(10, sc_float32, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((float)(i * 3), sc_float32);
        sc_set_vector_element(vector, i, val);
    }

    sc_value_t scalar = to_sc_value((float)2, sc_float32);
    sc_vector* result = sc_vector_sub_scalar(vector, scalar, arena);
    if (!result) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (int32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        if (val.type != sc_float32 || val.value.f32 != (float)(i * 3 - 2)) {
            CCB_WARNING("Vector-scalar subtraction mismatch at index %u: expected %f, got %f", i, (float)(i * 3 - 2), val.value.f32);
            return -1;
        }
    }

    return 0;
}

int test_vector_sub_scalar_inplace_float(ccb_arena* arena) {
    sc_vector* vector = sc_create_vector(10, sc_float32, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((float)(i * 3), sc_float32);
        sc_set_vector_element(vector, i, val);
    }

    sc_value_t scalar = to_sc_value((float)2, sc_float32);
    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(vector, i);
        val.value.f32 -= scalar.value.f32;
        sc_set_vector_element(vector, i, val);
    }

    for (int32_t i = 0; i < 10; i++) {
         sc_value_t val = sc_get_vector_element(vector, i);
         if (val.type != sc_float32 || val.value.f32 != (float)(i * 3 - 2)) {
             CCB_WARNING("Vector-scalar subtraction mismatch at index %u: expected %f, got %f", i, (float)(i * 3 - 2), val.value.f32);
             return -1;
         }
    }

    return 0;
}

int test_vector_mul_ellement_wise_float(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float32, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float32, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector2");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((float)i, sc_float32);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((float)(i * 2), sc_float32);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_vector* result = sc_vector_mul_ellement_wise(vector1, vector2, arena);
    if (!result) {
        CCB_WARNING("Failed to create result vectore");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        if (val.type != sc_float32 || val.value.f32 != (float)(i * i * 2)) {
            CCB_WARNING("Vector element-wise multiplication mismatch at index %u: expected %f, got %f", i, (float)(i * i * 2), val.value.f32);
            return -1;
        }
    }

    return 0;
}

int test_vector_mul_ellement_wise_inplace_float(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float32, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float32, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector2");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((float)i, sc_float32);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((float)(i * 2), sc_float32);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_vector* result = sc_vector_mul_ellement_wise_inplace(vector1, vector2);
    if (!result) {
        CCB_WARNING("Failed to create result vectore");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        if (val.type != sc_float32 || val.value.f32 != (float)(i * i * 2)) {
            CCB_WARNING("Vector element-wise multiplication mismatch at index %u: expected %f, got %f", i, (float)(i * i * 2), val.value.f32);
            return -1;
        }
    }

    return 0;
}

int test_vector_mul_scalar_float(ccb_arena* arena) {
    sc_vector* vector = sc_create_vector(10, sc_float32, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((float)i, sc_float32);
        sc_set_vector_element(vector, i, val);
    }

    sc_value_t scalar = to_sc_value((float)2, sc_float32);
    sc_vector* result = sc_vector_mul_scalar(vector, scalar, arena);
    if (!result){
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        if (val.type != sc_float32 || val.value.f32 != (float)(2 * i)) {
            CCB_WARNING("Vector-scalar multiplication mismatch at index %u: expected %f, got %f", i, (float)(2 * i), val.value.f32);
            return -1;
        }
    }

    return 0;
}

int test_vector_mul_scalar_inplace_float(ccb_arena* arena) {
    sc_vector* vector = sc_create_vector(10, sc_float32, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((float)i, sc_float32);
        sc_set_vector_element(vector, i, val);
    }

    sc_value_t scalar = to_sc_value((float)2, sc_float32);
    sc_vector* result = sc_vector_mul_scalar_inplace(vector, scalar);
    if (!result){
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        if (val.type != sc_float32 || val.value.f32 != (float)(2 * i)) {
            CCB_WARNING("Vector-scalar multiplication mismatch at index %u: expected %f, got %f", i, (float)(2 * i), val.value.f32);
            return -1;
        }
    }

    return 0;
}

int test_vector_div_element_wise_float(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float32, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float32, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector2");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((float)i, sc_float32);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((float)(i+1), sc_float32);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_vector* result = sc_vector_div_ellement_wise(vector1, vector2, arena);
    if (!result) {
        CCB_WARNING("Failed to create result vectore");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        float expected = (float)i / (float)(i +1);
        if (val.type != sc_float32 || val.value.f32 != expected) {
            CCB_WARNING("Vector element-wise multiplication mismatch at index %u: expected %f, got %f", i, expected, (float)val.value.f32);
            return -1;
        }
    }

    return 0;
}

int test_vector_div_element_wise_inplace_float(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float32, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float32, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector2");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((float)i, sc_float32);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((float)(i+1), sc_float32);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_vector* result = sc_vector_div_ellement_wise_inplace(vector1, vector2);
    if (!result) {
        CCB_WARNING("Failed to create result vectore");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        float expected = (float)i / (float)(i +1);
        if (val.type != sc_float32 || val.value.f32 != expected) {
            CCB_WARNING("Vector element-wise multiplication mismatch at index %u: expected %f, got %f", i, expected, (float)val.value.f32);
            return -1;
        }
    }

    return 0;
}

int test_vector_div_scalar_float(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float32, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((float)i, sc_float32);
        sc_set_vector_element(vector1, i, val);
    }

    sc_value_t scalar = to_sc_value((float)2, sc_float32);
    sc_vector* result = sc_vector_div_scalar(vector1, scalar, arena);
    if (!result) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        float expected = (float)i / (float)2;
        if (val.type != sc_float32 || val.value.f32 != expected) {
            CCB_WARNING("Vector-scalar division mismatch at index %u: expected %f, got %f", i, expected, (float)val.value.f32);
            return -1;
        }
    }

    return 0;
}

int test_vector_div_scalar_inplace_float(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float32, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((float)i, sc_float32);
        sc_set_vector_element(vector1, i, val);
    }

    sc_value_t scalar = to_sc_value((float)2, sc_float32);
    sc_vector* result = sc_vector_div_scalar_inplace(vector1, scalar);
    if (!result) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        float expected = (float)i / (float)2;
        if (val.type != sc_float32 || val.value.f32 != expected) {
            CCB_WARNING("Vector-scalar division mismatch at index %u: expected %f, got %f", i, expected, (float)val.value.f32);
            return -1;
        }
    }

    return 0;
}

int test_vector_dot_product_float(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float32, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float32, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((float)i/2, sc_float32);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((float)(i * 2), sc_float32);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_value_t result = sc_vector_dot(vector1, vector2);

    float expected = 0;
    for (uint32_t i = 0; i < 10; i++) {
        expected += (float)i/2 * (float)(i * 2);
    }

    if (result.type != sc_float32 || result.value.f32 != expected) {
        CCB_WARNING("Vector dot product mismatch: expected %f, got %f", (float)expected, (float)result.value.f32);
        return -1;
    }

    return 0;
}

int test_vector_cross_product_float(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(3, sc_float32, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(3, sc_float32, arena);
    if (!vector2){
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    for (uint32_t i = 0; i < 3; i++) {
        sc_value_t val1 = to_sc_value((float)i, sc_float32);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((float)(i + 1), sc_float32);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_vector* result = sc_vector_cross(vector1, vector2, arena);
    if (!result) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    sc_vector* expected = sc_create_vector(3, sc_float32, arena);
    if (!expected) {
        CCB_WARNING("Failed to create expected vector");
        return -1;
    }

    float* a = (float*)vector1->data;
    float* b = (float*)vector2->data;
    float* c = (float*)expected->data;
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];

    for (uint32_t i = 0; i < 3; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        sc_value_t expected_val = sc_get_vector_element(expected, i);
        if (val.type != sc_float32 || val.value.f32 != expected_val.value.f32) {
            CCB_WARNING("Vector dot product mismatch: expected %f, got %f", (float)expected_val.value.f32, (float)val.value.f32);
            return -1;
         }
    }

    return 0;
}

int test_vector_cross_product_inplace_float(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(3, sc_float32, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(3, sc_float32, arena);
    if (!vector2){
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    for (uint32_t i = 0; i < 3; i++) {
        sc_value_t val1 = to_sc_value((float)i, sc_float32);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((float)(i + 1), sc_float32);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_vector* expected = sc_create_vector(3, sc_float32, arena);
    if (!expected) {
        CCB_WARNING("Failed to create expected vector");
        return -1;
    }

    float* a = (float*)vector1->data;
    float* b = (float*)vector2->data;
    float* c = (float*)expected->data;
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];

    sc_vector* result = sc_vector_cross_inplace(vector1, vector2);
    if (!result) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 3; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        sc_value_t expected_val = sc_get_vector_element(expected, i);
        if (val.type != sc_float32 || val.value.f32 != expected_val.value.f32) {
            CCB_WARNING("Vector dot product mismatch: expected %f, got %f", (float)expected_val.value.f32, (float)val.value.f32);
            return -1;
         }
    }

    return 0;
}

int test_vector_norm1_float(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float32, arena);
    CCB_NOTNULL(vector1, "Failed to create vector1");

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((float)i, sc_float32);
        sc_set_vector_element(vector1, i, val);
    }

    sc_value_t result = sc_vector_norm(vector1, 1, arena);
    if (result.type != sc_float32) {
        CCB_WARNING("Vector norm mismatch: expected %f, got %f", (float)0, (float)result.value.f32);
        return -1;
    }

    sc_value_t expected = to_sc_value((float)0, sc_float32);
    for (uint32_t i = 0; i < 10; i++) {
        expected = sc_scalar_add(expected, sc_scalar_abs(sc_get_vector_element(vector1, i)));
    }

    if (result.value.f32 != expected.value.f32) {
         CCB_WARNING("Vector norm mismatch: expected %f, got %f", (float)expected.value.f32, (float)result.value.f32);
         return -1;
    }

    return 0;
}

int test_vector_norm2_float(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float32, arena);
    CCB_NOTNULL(vector1, "Failed to create vector1");

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((float)i, sc_float32);
        sc_set_vector_element(vector1, i, val);
    }

    sc_value_t result = sc_vector_norm(vector1, 2, arena);
    if (result.type != sc_float32) {
        CCB_WARNING("Vector norm mismatch: expected %f, got %f", (float)0, (float)result.value.f32);
        return -1;
    }

    sc_value_t expected = to_sc_value((float)0, sc_float32);
    sc_value_t val_p = to_sc_value((float)2, sc_float32);
    for (uint32_t i = 0; i < 10; i++) {
        expected = sc_scalar_add(expected, sc_scalar_pow(sc_get_vector_element(vector1, i), val_p));
    }

    expected = sc_scalar_root(expected, val_p);
    if (result.value.f32 != expected.value.f32) {
         CCB_WARNING("Vector norm mismatch: expected %f, got %f", (float)expected.value.f32, (float)result.value.f32);
         return -1;
    }

    return 0;
}

int test_vector_norm3_float(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float32, arena);
    CCB_NOTNULL(vector1, "Failed to create vector1");

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((float)i, sc_float32);
        sc_set_vector_element(vector1, i, val);
    }

    sc_value_t result = sc_vector_norm(vector1, 3, arena);
    if (result.type != sc_float32) {
        CCB_WARNING("Vector norm mismatch: expected %f, got %f", (float)0, (float)result.value.f32);
        return -1;
    }

    sc_value_t expected = to_sc_value((float)0, sc_float32);
    sc_value_t val_p = to_sc_value((float)3, sc_float32);
    for (uint32_t i = 0; i < 10; i++) {
        expected = sc_scalar_add(expected, sc_scalar_pow(sc_get_vector_element(vector1, i), val_p));
    }

    expected = sc_scalar_root(expected, val_p);
    if (result.value.f32 != expected.value.f32) {
         CCB_WARNING("Vector norm mismatch: expected %f, got %f", (float)expected.value.f32, (float)result.value.f32);
         return -1;
    }

    return 0;
}

int test_vector_normalization_float(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float32, arena);
    CCB_NOTNULL(vector1, "Failed to create vector1");

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((float)i, sc_float32);
        sc_set_vector_element(vector1, i, val);
    }

    sc_vector* result = sc_vector_normalize(vector1, arena);
    if (!result) {
         CCB_WARNING("Failed to create result vector");
         return -1;
    }

    sc_value_t norm = sc_vector_norm(vector1, 2, arena);
    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        sc_value_t expected = sc_scalar_div(sc_get_vector_element(vector1, i), norm);
        if (val.type != sc_float32 || val.value.f32 != expected.value.f32) {
            CCB_WARNING("Vector normalization mismatch at index %u: expected %f, got %f", i, (float)expected.value.f32, (float)val.value.f32);
            return -1;
        }
    }

    return 0;
}

int test_vector_normalization_inplace_float(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float32, arena);
    CCB_NOTNULL(vector1, "Failed to create vector1");

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((float)i, sc_float32);
        sc_set_vector_element(vector1, i, val);
    }

    sc_value_t norm = sc_vector_norm(vector1, 2, arena);
    sc_vector* result = sc_vector_normalize_inplace(vector1, arena);
    if (!result) {
         CCB_WARNING("Failed to create result vector");
         return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        sc_value_t expected = sc_scalar_div(to_sc_value((float)i, sc_float32), norm);
        if (val.type != sc_float32 || val.value.f32 != expected.value.f32) {
            CCB_WARNING("Vector normalization mismatch at index %u: expected %f, got %f", i, (float)expected.value.f32, (float)val.value.f32);
            return -1;
        }
    }

    return 0;
}

int test_get_sub_tensor_float(ccb_arena* arena) {
    uint32_t shape[3] = {4, 4, 4};
    sc_dimensions* dims = sc_create_dimensions(3, arena, shape);
    sc_tensor* tensor = sc_create_tensor(dims, sc_float32, arena);
    CCB_NOTNULL(tensor, "Failed to create tensor");

    sc_index* zero_idx = sc_create_index(3, arena, (uint32_t[]){0, 0, 0});
    CCB_NOTNULL(zero_idx, "Failed to create zero index");

    for (uint32_t i = 0; i < 64; i++) {
        sc_value_t val = to_sc_value((float)i, sc_float32);
        for (uint32_t j = 0; j < 3; j++) {
            zero_idx->indices[j] = (i / (uint32_t)pow(4, 2 - j)) % 4;
        }
        sc_set_tensor_element(tensor, zero_idx, val);
    }

    uint32_t start[3] = {1, 1};
    sc_index* indices = sc_create_index(2, arena, start);
    CCB_NOTNULL(indices, "Failed to create indices");

    sc_tensor* sub_tensor = sc_get_sub_tensor(tensor, indices, arena);
    CCB_NOTNULL(sub_tensor, "Failed to create sub-tensor");

    CCB_INFO("Sub-tensor allocated: %p", sub_tensor);
    uint32_t expected_shape[1] = {4};
    if (sub_tensor->dims->dims_count != 1 || memcmp(sub_tensor->dims->dims, expected_shape, sizeof(expected_shape)) != 0) {
        CCB_WARNING("Sub-tensor shape mismatch: expected [4], got [%u, %u, %u]", sub_tensor->dims->dims[0], sub_tensor->dims->dims[1], sub_tensor->dims->dims[2]);
        return -1;
    }

    for (uint32_t i = 0; i < 4; i++) {
        sc_index* idx = sc_create_index(1, arena, &i);
        sc_value_t val = sc_get_tensor_element(sub_tensor, idx);
        float expected = (float)(16 + i);
        if (val.type != sc_float32 || val.value.f32 != expected) {
            CCB_WARNING("Sub-tensor element mismatch at index %u: expected %f, got %f", i, (float)expected, (float)val.value.f32);
            return -1;
        }
    }

    return 0;
}

int test_vector_creation_double(ccb_arena* arena){
    sc_vector* vector = sc_create_vector(10, sc_float64, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }
    return 0;
}

int test_tensor_creation_double(ccb_arena* arena){
    sc_dimensions* dims = sc_create_dimensions(2, arena, (uint32_t[]){2, 5});
    if (!dims) {
        CCB_WARNING("Failed to create dimensions");
        return -1;
    }

    sc_tensor* tensor = sc_create_tensor(dims, sc_float64, arena);
    if (!tensor) {
        CCB_WARNING("Failed to create tensor");
        return -1;
    }
    return 0;
}

int test_vector_data_loading_double(ccb_arena* arena){
    sc_vector* vector = sc_create_vector(10, sc_float64, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    double data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    sc_data_to_vector(vector, data, 10);
    return 0;
}

int test_tensor_data_loading_double(ccb_arena* arena){
    sc_dimensions* dims = sc_create_dimensions(2, arena, (uint32_t[]){2, 5});
    if (!dims) {
        CCB_WARNING("Failed to create dimensions");
        return -1;
    }

    sc_tensor* tensor = sc_create_tensor(dims, sc_float64, arena);
    if (!tensor) {
        CCB_WARNING("Failed to create tensor");
        return -1;
    }

    double data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    sc_data_to_tensor(tensor, data, 10);
    return 0;
}

int test_vector_clone_double(ccb_arena* arena) {
    sc_vector* vector = sc_create_vector(10, sc_float64, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((double)i, sc_float64);
        sc_set_vector_element(vector, i, val);
    }

    sc_vector* clone = sc_clone_vector(vector, arena);
    if (!clone) {
        CCB_WARNING("Failed to clone vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t original_val = sc_get_vector_element(vector, i);
        sc_value_t clone_val = sc_get_vector_element(clone, i);
        if (original_val.value.f64 != clone_val.value.f64) {
            CCB_WARNING("Vector clone failed");
            return -1;
        }
    }

    return 0;
}

int test_tensor_clone_double(ccb_arena* arena) {
    sc_dimensions* dims = sc_create_dimensions(2, arena, (uint32_t[]){3, 4});
    if (!dims) {
        CCB_WARNING("Failed to create dimensions");
        return -1;
    }

    sc_tensor* tensor = sc_create_tensor(dims, sc_float64, arena);
    if (!tensor) {
        CCB_WARNING("Failed to create tensor");
        return -1;
    }

    for (uint32_t i = 0; i < 3; i++) {
        for (uint32_t j = 0; j < 4; j++) {
            sc_index index;
            index.count = 2;
            uint32_t idxs[2] = {i, j};
            index.indices = idxs;
            sc_value_t val = to_sc_value((double)(i * 4 + j), sc_float64);
            sc_set_tensor_element(tensor, &index, val);
        }
    }

    sc_tensor* clone = sc_clone_tensor(tensor, arena);
    if (!clone) {
        CCB_WARNING("Failed to clone tensor");
        return -1;
    }

    for (uint32_t i = 0; i < 3; i++) {
        for (uint32_t j = 0; j < 4; j++) {
            sc_index index;
            index.count = 2;
            uint32_t idxs[2] = {i, j};
            index.indices = idxs;

            sc_value_t original_val = sc_get_tensor_element(tensor, &index);
            sc_value_t clone_val = sc_get_tensor_element(clone, &index);
            if (original_val.value.f64 != clone_val.value.f64) {
                CCB_WARNING("tensor clone failed");
                return -1;
            }
        }
    }

    return 0;
}

int test_vector_set_get_double(ccb_arena* arena) {
    sc_vector* vector = sc_create_vector(10, sc_float64, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((double)i, sc_float64);
        sc_set_vector_element(vector, i, val);
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(vector, i);
        if (val.type != sc_float64 || val.value.f64 != (double)i) {
            CCB_WARNING("Vector set/get mismatch at index %u: expected %f, got %f", i, (double)i, val.value.f64);
            return -1;
        }
    }

    return 0;
}

int test_tensor_set_get_double(ccb_arena* arena) {
    sc_dimensions* dims = sc_create_dimensions(2, arena, (uint32_t[]){2, 5});
    if (!dims) {
        CCB_WARNING("Failed to create dimensions");
        return -1;
    }

    sc_tensor* tensor = sc_create_tensor(dims, sc_float64, arena);
    if (!tensor) {
        CCB_WARNING("Failed to create tensor");
        return -1;
    }

    for (uint32_t i = 0; i < 2; i++) {
        for (uint32_t j = 0; j < 5; j++) {
            sc_index index;
            index.count = 2;
            uint32_t idxs[2] = {i, j};
            index.indices = idxs;
            sc_value_t val = to_sc_value((double)(i * 5 + j) * 3.0, sc_float64);
            sc_set_tensor_element(tensor, &index, val);
        }
    }

    for (uint32_t i = 0; i < 2; i++) {
        for (uint32_t j = 0; j < 5; j++) {
            sc_index index;
            index.count = 2;
            uint32_t idxs[2] = {i, j};
            index.indices = idxs;

            sc_value_t val = sc_get_tensor_element(tensor, &index);
            if (val.type != sc_float64 || val.value.f64 != ((double)(i * 5 + j) * 3.0)) {
                CCB_WARNING("tensor set/get mismatch at index [%u, %u]: expected %f, got %f", i, j, (double)(i * 5 + j) * 3.0, val.value.f64);
                return -1;
            }
        }
    }

    return 0;
}

int test_vector_add_double(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float64, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float64, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector2");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((double)i, sc_float64);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((double)(i * 2), sc_float64);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_vector* result = sc_vector_add(vector1, vector2, arena);
    if (!result) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        if (val.type != sc_float64 || val.value.f64 != (double)(i + i * 2)) {
            CCB_WARNING("Vector addition mismatch at index %u: expected %f, got %f", i, (double)(i + i * 2), val.value.f64);
            return -1;
        }
    }

    return 0;
}

int test_vector_add_inplace_double(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float64, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float64, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector2");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((double)i, sc_float64);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((double)(i * 2), sc_float64);
        sc_set_vector_element(vector2, i, val2);
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = sc_get_vector_element(vector1, i);
        sc_value_t val2 = sc_get_vector_element(vector2, i);
        sc_value_t sum;
        sum.type = sc_float64;
        sum.value.f64 = val1.value.f64 + val2.value.f64;
        sc_set_vector_element(vector1, i, sum);
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(vector1, i);
        if (val.type != sc_float64 || val.value.f64 != (double)(i + i * 2)) {
            CCB_WARNING("In-place vector addition mismatch at index %u : expected %f, got %f", i, (double)(i + i * 2), val.value.f64);
            return -1;
        }
    }

    return 0;
}

int test_vector_add_scalar_double(ccb_arena* arena) {
    sc_vector* vector = sc_create_vector(10, sc_float64, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((double)i, sc_float64);
        sc_set_vector_element(vector, i, val);
    }

    sc_value_t scalar = to_sc_value((double)5, sc_float64);
    sc_vector* result = sc_vector_add_scalar(vector, scalar, arena);
    if (!result) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        if (val.type != sc_float64 || val.value.f64 != (double)(i + 5)) {
            CCB_WARNING("Vector-scalar addition mismatch at index %u   : expected %f, got %f", i, (double)(i + 5), val.value.f64);
            return -1;
        }
    }

    return 0;
}

int test_vector_add_scalar_inplace_double(ccb_arena* arena) {
    sc_vector* vector = sc_create_vector(10, sc_float64, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((double)i, sc_float64);
        sc_set_vector_element(vector, i, val);
    }

    sc_value_t scalar = to_sc_value((double)5, sc_float64);
    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(vector, i);
        sc_value_t sum;
        sum.type = sc_float64;
        sum.value.f64 = val.value.f64 + scalar.value.f64;
        sc_set_vector_element(vector, i, sum);
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(vector, i);
        if (val.type != sc_float64 || val.value.f64 != (double)(i + 5)) {
            CCB_WARNING("In-place vector-scalar addition mismatch at index %u : expected %f, got %f", i, (double)(i + 5), val.value.f64);
            return -1;
        }
    }

    return 0;
}

int test_vector_sub_double(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float64, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float64, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector2");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((double)(i * 3), sc_float64);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((double)(i * 2), sc_float64);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_vector* result = sc_vector_sub(vector1, vector2, arena);
    if (!result) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        if (val.type != sc_float64 || val.value.f64 != (double)(i * 3 - i * 2)) {
            CCB_WARNING("Vector subtraction mismatch at index %u: expected %f, got %f", i, (double)(i * 3 - i * 2), val.value.f64);
            return -1;
        }
    }

    return 0;
}

int test_vector_sub_inplace_double(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float64, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float64, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector2");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((double)(i * 3), sc_float64);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((double)(i * 2), sc_float64);
        sc_set_vector_element(vector2, i, val2);
    }

    vector1 = sc_vector_sub_inplace(vector1, vector2);
    if (!vector1) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(vector1, i);
        if (val.type != sc_float64 || val.value.f64 != (double)(i * 3 - i * 2)) {
            CCB_WARNING("Vector subtraction mismatch at index %u: expected %f, got %f", i, (double)(i * 3 - i * 2), val.value.f64);
            return -1;
        }
    }

    return 0;
}

int test_vector_sub_scalar_double(ccb_arena* arena) {
    sc_vector* vector = sc_create_vector(10, sc_float64, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((double)(i * 3), sc_float64);
        sc_set_vector_element(vector, i, val);
    }

    sc_value_t scalar = to_sc_value((double)2, sc_float64);
    sc_vector* result = sc_vector_sub_scalar(vector, scalar, arena);
    if (!result) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (int32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        if (val.type != sc_float64 || val.value.f64 != (double)(i * 3 - 2)) {
            CCB_WARNING("Vector-scalar subtraction mismatch at index %u: expected %f, got %f", i, (double)(i * 3 - 2), val.value.f64);
            return -1;
        }
    }

    return 0;
}

int test_vector_sub_scalar_inplace_double(ccb_arena* arena) {
    sc_vector* vector = sc_create_vector(10, sc_float64, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((double)(i * 3), sc_float64);
        sc_set_vector_element(vector, i, val);
    }

    sc_value_t scalar = to_sc_value((double)2, sc_float64);
    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(vector, i);
        val.value.f64 -= scalar.value.f64;
        sc_set_vector_element(vector, i, val);
    }

    for (int32_t i = 0; i < 10; i++) {
         sc_value_t val = sc_get_vector_element(vector, i);
         if (val.type != sc_float64 || val.value.f64 != (double)(i * 3 - 2)) {
             CCB_WARNING("Vector-scalar subtraction mismatch at index %u: expected %f, got %f", i, (double)(i * 3 - 2), val.value.f64);
             return -1;
         }
    }

    return 0;
}

int test_vector_mul_ellement_wise_double(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float64, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float64, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector2");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((double)i, sc_float64);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((double)(i * 2), sc_float64);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_vector* result = sc_vector_mul_ellement_wise(vector1, vector2, arena);
    if (!result) {
        CCB_WARNING("Failed to create result vectore");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        if (val.type != sc_float64 || val.value.f64 != (double)(i * i * 2)) {
            CCB_WARNING("Vector element-wise multiplication mismatch at index %u: expected %f, got %f", i, (double)(i * i * 2), val.value.f64);
            return -1;
        }
    }

    return 0;
}

int test_vector_mul_ellement_wise_inplace_double(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float64, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float64, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector2");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((double)i, sc_float64);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((double)(i * 2), sc_float64);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_vector* result = sc_vector_mul_ellement_wise_inplace(vector1, vector2);
    if (!result) {
        CCB_WARNING("Failed to create result vectore");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        if (val.type != sc_float64 || val.value.f64 != (double)(i * i * 2)) {
            CCB_WARNING("Vector element-wise multiplication mismatch at index %u: expected %f, got %f", i, (double)(i * i * 2), val.value.f64);
            return -1;
        }
    }

    return 0;
}

int test_vector_mul_scalar_double(ccb_arena* arena) {
    sc_vector* vector = sc_create_vector(10, sc_float64, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((double)i, sc_float64);
        sc_set_vector_element(vector, i, val);
    }

    sc_value_t scalar = to_sc_value((double)2, sc_float64);
    sc_vector* result = sc_vector_mul_scalar(vector, scalar, arena);
    if (!result){
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        if (val.type != sc_float64 || val.value.f64 != (double)(2 * i)) {
            CCB_WARNING("Vector-scalar multiplication mismatch at index %u: expected %f, got %f", i, (double)(2 * i), val.value.f64);
            return -1;
        }
    }

    return 0;
}

int test_vector_mul_scalar_inplace_double(ccb_arena* arena) {
    sc_vector* vector = sc_create_vector(10, sc_float64, arena);
    if (!vector) {
        CCB_WARNING("Failed to create vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((double)i, sc_float64);
        sc_set_vector_element(vector, i, val);
    }

    sc_value_t scalar = to_sc_value((double)2, sc_float64);
    sc_vector* result = sc_vector_mul_scalar_inplace(vector, scalar);
    if (!result){
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        if (val.type != sc_float64 || val.value.f64 != (double)(2 * i)) {
            CCB_WARNING("Vector-scalar multiplication mismatch at index %u: expected %f, got %f", i, (double)(2 * i), val.value.f64);
            return -1;
        }
    }

    return 0;
}

int test_vector_div_element_wise_double(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float64, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float64, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector2");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((double)i, sc_float64);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((double)(i+1), sc_float64);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_vector* result = sc_vector_div_ellement_wise(vector1, vector2, arena);
    if (!result) {
        CCB_WARNING("Failed to create result vectore");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        double expected = (double)i / (double)(i +1);
        if (val.type != sc_float64 || val.value.f64 != expected) {
            CCB_WARNING("Vector element-wise multiplication mismatch at index %u: expected %f, got %f", i, expected, (float)val.value.f64);
            return -1;
        }
    }

    return 0;
}

int test_vector_div_element_wise_inplace_double(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float64, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float64, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector2");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((double)i, sc_float64);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((double)(i+1), sc_float64);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_vector* result = sc_vector_div_ellement_wise_inplace(vector1, vector2);
    if (!result) {
        CCB_WARNING("Failed to create result vectore");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        double expected = (double)i / (double)(i +1);
        if (val.type != sc_float64 || val.value.f64 != expected) {
            CCB_WARNING("Vector element-wise multiplication mismatch at index %u: expected %f, got %f", i, expected, (float)val.value.f64);
            return -1;
        }
    }

    return 0;
}

int test_vector_div_scalar_double(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float64, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((double)i, sc_float64);
        sc_set_vector_element(vector1, i, val);
    }

    sc_value_t scalar = to_sc_value((double)2, sc_float64);
    sc_vector* result = sc_vector_div_scalar(vector1, scalar, arena);
    if (!result) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        double expected = (double)i / (double)2;
        if (val.type != sc_float64 || val.value.f64 != expected) {
            CCB_WARNING("Vector-scalar division mismatch at index %u: expected %f, got %f", i, expected, (float)val.value.f64);
            return -1;
        }
    }

    return 0;
}

int test_vector_div_scalar_inplace_double(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float64, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((double)i, sc_float64);
        sc_set_vector_element(vector1, i, val);
    }

    sc_value_t scalar = to_sc_value((double)2, sc_float64);
    sc_vector* result = sc_vector_div_scalar_inplace(vector1, scalar);
    if (!result) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        double expected = (double)i / (double)2;
        if (val.type != sc_float64 || val.value.f64 != expected) {
            CCB_WARNING("Vector-scalar division mismatch at index %u: expected %f, got %f", i, expected, (float)val.value.f64);
            return -1;
        }
    }

    return 0;
}

int test_vector_dot_product_double(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float64, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(10, sc_float64, arena);
    if (!vector2) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val1 = to_sc_value((double)i/2, sc_float64);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((double)(i * 2), sc_float64);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_value_t result = sc_vector_dot(vector1, vector2);

    double expected = 0;
    for (uint32_t i = 0; i < 10; i++) {
        expected += (double)i/2 * (double)(i * 2);
    }

    if (result.type != sc_float64 || result.value.f64 != expected) {
        CCB_WARNING("Vector dot product mismatch: expected %f, got %f", (float)expected, (float)result.value.f64);
        return -1;
    }

    return 0;
}

int test_vector_cross_product_double(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(3, sc_float64, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(3, sc_float64, arena);
    if (!vector2){
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    for (uint32_t i = 0; i < 3; i++) {
        sc_value_t val1 = to_sc_value((double)i, sc_float64);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((double)(i + 1), sc_float64);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_vector* result = sc_vector_cross(vector1, vector2, arena);
    if (!result) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    sc_vector* expected = sc_create_vector(3, sc_float64, arena);
    if (!expected) {
        CCB_WARNING("Failed to create expected vector");
        return -1;
    }

    double* a = (double*)vector1->data;
    double* b = (double*)vector2->data;
    double* c = (double*)expected->data;
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];

    for (uint32_t i = 0; i < 3; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        sc_value_t expected_val = sc_get_vector_element(expected, i);
        if (val.type != sc_float64 || val.value.f64 != expected_val.value.f64) {
            CCB_WARNING("Vector dot product mismatch: expected %f, got %f", (float)expected_val.value.f64, (float)val.value.f64);
            return -1;
         }
    }

    return 0;
}

int test_vector_cross_product_inplace_double(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(3, sc_float64, arena);
    if (!vector1) {
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    sc_vector* vector2 = sc_create_vector(3, sc_float64, arena);
    if (!vector2){
        CCB_WARNING("Failed to create vector1");
        return -1;
    }

    for (uint32_t i = 0; i < 3; i++) {
        sc_value_t val1 = to_sc_value((double)i, sc_float64);
        sc_set_vector_element(vector1, i, val1);
        sc_value_t val2 = to_sc_value((double)(i + 1), sc_float64);
        sc_set_vector_element(vector2, i, val2);
    }

    sc_vector* expected = sc_create_vector(3, sc_float64, arena);
    if (!expected) {
        CCB_WARNING("Failed to create expected vector");
        return -1;
    }

    double* a = (double*)vector1->data;
    double* b = (double*)vector2->data;
    double* c = (double*)expected->data;
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];

    sc_vector* result = sc_vector_cross_inplace(vector1, vector2);
    if (!result) {
        CCB_WARNING("Failed to create result vector");
        return -1;
    }

    for (uint32_t i = 0; i < 3; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        sc_value_t expected_val = sc_get_vector_element(expected, i);
        if (val.type != sc_float64 || val.value.f64 != expected_val.value.f64) {
            CCB_WARNING("Vector dot product mismatch: expected %f, got %f", (float)expected_val.value.f64, (float)val.value.f64);
            return -1;
         }
    }

    return 0;
}

int test_vector_norm1_double(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float64, arena);
    CCB_NOTNULL(vector1, "Failed to create vector1");

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((double)i, sc_float64);
        sc_set_vector_element(vector1, i, val);
    }

    sc_value_t result = sc_vector_norm(vector1, 1, arena);
    if (result.type != sc_float64) {
        CCB_WARNING("Vector norm mismatch: expected %f, got %f", (float)0, (float)result.value.f64);
        return -1;
    }

    sc_value_t expected = to_sc_value((double)0, sc_float64);
    for (uint32_t i = 0; i < 10; i++) {
        expected = sc_scalar_add(expected, sc_scalar_abs(sc_get_vector_element(vector1, i)));
    }

    if (result.value.f64 != expected.value.f64) {
         CCB_WARNING("Vector norm mismatch: expected %f, got %f", (float)expected.value.f64, (float)result.value.f64);
         return -1;
    }

    return 0;
}

int test_vector_norm2_double(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float64, arena);
    CCB_NOTNULL(vector1, "Failed to create vector1");

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((double)i, sc_float64);
        sc_set_vector_element(vector1, i, val);
    }

    sc_value_t result = sc_vector_norm(vector1, 2, arena);
    if (result.type != sc_float64) {
        CCB_WARNING("Vector norm mismatch: expected %f, got %f", (float)0, (float)result.value.f64);
        return -1;
    }

    sc_value_t expected = to_sc_value((double)0, sc_float64);
    sc_value_t val_p = to_sc_value((double)2, sc_float64);
    for (uint32_t i = 0; i < 10; i++) {
        expected = sc_scalar_add(expected, sc_scalar_pow(sc_get_vector_element(vector1, i), val_p));
    }

    expected = sc_scalar_root(expected, val_p);
    if (result.value.f64 != expected.value.f64) {
         CCB_WARNING("Vector norm mismatch: expected %f, got %f", (float)expected.value.f64, (float)result.value.f64);
         return -1;
    }

    return 0;
}

int test_vector_norm3_double(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float64, arena);
    CCB_NOTNULL(vector1, "Failed to create vector1");

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((double)i, sc_float64);
        sc_set_vector_element(vector1, i, val);
    }

    sc_value_t result = sc_vector_norm(vector1, 3, arena);
    if (result.type != sc_float64) {
        CCB_WARNING("Vector norm mismatch: expected %f, got %f", (float)0, (float)result.value.f64);
        return -1;
    }

    sc_value_t expected = to_sc_value((double)0, sc_float64);
    sc_value_t val_p = to_sc_value((double)3, sc_float64);
    for (uint32_t i = 0; i < 10; i++) {
        expected = sc_scalar_add(expected, sc_scalar_pow(sc_get_vector_element(vector1, i), val_p));
    }

    expected = sc_scalar_root(expected, val_p);
    if (result.value.f64 != expected.value.f64) {
         CCB_WARNING("Vector norm mismatch: expected %f, got %f", (float)expected.value.f64, (float)result.value.f64);
         return -1;
    }

    return 0;
}

int test_vector_normalization_double(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float64, arena);
    CCB_NOTNULL(vector1, "Failed to create vector1");

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((double)i, sc_float64);
        sc_set_vector_element(vector1, i, val);
    }

    sc_vector* result = sc_vector_normalize(vector1, arena);
    if (!result) {
         CCB_WARNING("Failed to create result vector");
         return -1;
    }

    sc_value_t norm = sc_vector_norm(vector1, 2, arena);
    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        sc_value_t expected = sc_scalar_div(sc_get_vector_element(vector1, i), norm);
        if (val.type != sc_float64 || val.value.f64 != expected.value.f64) {
            CCB_WARNING("Vector normalization mismatch at index %u: expected %f, got %f", i, (float)expected.value.f64, (float)val.value.f64);
            return -1;
        }
    }

    return 0;
}

int test_vector_normalization_inplace_double(ccb_arena* arena) {
    sc_vector* vector1 = sc_create_vector(10, sc_float64, arena);
    CCB_NOTNULL(vector1, "Failed to create vector1");

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = to_sc_value((double)i, sc_float64);
        sc_set_vector_element(vector1, i, val);
    }

    sc_value_t norm = sc_vector_norm(vector1, 2, arena);
    sc_vector* result = sc_vector_normalize_inplace(vector1, arena);
    if (!result) {
         CCB_WARNING("Failed to create result vector");
         return -1;
    }

    for (uint32_t i = 0; i < 10; i++) {
        sc_value_t val = sc_get_vector_element(result, i);
        sc_value_t expected = sc_scalar_div(to_sc_value((double)i, sc_float64), norm);
        if (val.type != sc_float64 || val.value.f64 != expected.value.f64) {
            CCB_WARNING("Vector normalization mismatch at index %u: expected %f, got %f", i, (float)expected.value.f64, (float)val.value.f64);
            return -1;
        }
    }

    return 0;
}

int test_get_sub_tensor_double(ccb_arena* arena) {
    uint32_t shape[3] = {4, 4, 4};
    sc_dimensions* dims = sc_create_dimensions(3, arena, shape);
    sc_tensor* tensor = sc_create_tensor(dims, sc_float64, arena);
    CCB_NOTNULL(tensor, "Failed to create tensor");

    sc_index* zero_idx = sc_create_index(3, arena, (uint32_t[]){0, 0, 0});
    CCB_NOTNULL(zero_idx, "Failed to create zero index");

    for (uint32_t i = 0; i < 64; i++) {
        sc_value_t val = to_sc_value((double)i, sc_float64);
        for (uint32_t j = 0; j < 3; j++) {
            zero_idx->indices[j] = (i / (uint32_t)pow(4, 2 - j)) % 4;
        }
        sc_set_tensor_element(tensor, zero_idx, val);
    }

    uint32_t start[3] = {1, 1};
    sc_index* indices = sc_create_index(2, arena, start);
    CCB_NOTNULL(indices, "Failed to create indices");

    sc_tensor* sub_tensor = sc_get_sub_tensor(tensor, indices, arena);
    CCB_NOTNULL(sub_tensor, "Failed to create sub-tensor");

    CCB_INFO("Sub-tensor allocated: %p", sub_tensor);
    uint32_t expected_shape[1] = {4};
    if (sub_tensor->dims->dims_count != 1 || memcmp(sub_tensor->dims->dims, expected_shape, sizeof(expected_shape)) != 0) {
        CCB_WARNING("Sub-tensor shape mismatch: expected [4], got [%u, %u, %u]", sub_tensor->dims->dims[0], sub_tensor->dims->dims[1], sub_tensor->dims->dims[2]);
        return -1;
    }

    for (uint32_t i = 0; i < 4; i++) {
        sc_index* idx = sc_create_index(1, arena, &i);
        sc_value_t val = sc_get_tensor_element(sub_tensor, idx);
        double expected = (double)(16 + i);
        if (val.type != sc_float64 || val.value.f64 != expected) {
            CCB_WARNING("Sub-tensor element mismatch at index %u: expected %f, got %f", i, (float)expected, (float)val.value.f64);
            return -1;
        }
    }

    return 0;
}

int main(void) {
     ccb_arena* arena = ccb_init_arena();
     CCB_NOTNULL(arena, "Failed to initialize memory arena");

     int passed = 0;
     int failed = 0;
     int total = 0;

     ccb_InitLog("log\\test.log");

     if (test_dims_creation_(arena) == 0) {
         CCB_INFO("\e[32m[V] test_dims_creation_(null) passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_dims_creation_(null) failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_indices_creation_(arena) == 0) {
         CCB_INFO("\e[32m[V] test_indices_creation_(null) passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_indices_creation_(null) failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_slice_creation_(arena) == 0) {
         CCB_INFO("\e[32m[V] test_slice_creation_(null) passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_slice_creation_(null) failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_creation___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_creation___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_creation___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_tensor_creation___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_tensor_creation___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_tensor_creation___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_data_loading___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_data_loading___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_data_loading___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_tensor_data_loading___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_tensor_data_loading___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_tensor_data_loading___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_clone___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_clone___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_clone___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_tensor_clone___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_tensor_clone___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_tensor_clone___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_set_get___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_set_get___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_set_get___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_tensor_set_get___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_tensor_set_get___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_tensor_set_get___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_add___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_add___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_add___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_add_inplace___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_add_inplace___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_add_inplace___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_add_scalar___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_add_scalar___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_add_scalar___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_add_scalar_inplace___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_add_scalar_inplace___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_add_scalar_inplace___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_sub___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_sub___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_sub___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_sub_inplace___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_sub_inplace___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_sub_inplace___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_sub_scalar___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_sub_scalar___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_sub_scalar___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_sub_scalar_inplace___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_sub_scalar_inplace___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_sub_scalar_inplace___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_mul_ellement_wise___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_mul_ellement_wise___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_mul_ellement_wise___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_mul_ellement_wise_inplace___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_mul_ellement_wise_inplace___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_mul_ellement_wise_inplace___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_mul_scalar___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_mul_scalar___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_mul_scalar___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_mul_scalar_inplace___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_mul_scalar_inplace___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_mul_scalar_inplace___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_div_element_wise___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_div_element_wise___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_div_element_wise___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_div_element_wise_inplace___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_div_element_wise_inplace___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_div_element_wise_inplace___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_div_scalar___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_div_scalar___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_div_scalar___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_div_scalar_inplace___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_div_scalar_inplace___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_div_scalar_inplace___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_dot_product___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_dot_product___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_dot_product___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_cross_product___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_cross_product___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_cross_product___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_cross_product_inplace___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_cross_product_inplace___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_cross_product_inplace___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_norm1___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_norm1___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_norm1___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_norm2___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_norm2___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_norm2___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_norm3___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_norm3___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_norm3___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_normalization___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_normalization___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_normalization___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_normalization_inplace___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_normalization_inplace___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_normalization_inplace___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_get_sub_tensor___bf16(arena) == 0) {
         CCB_INFO("\e[32m[V] test_get_sub_tensor___bf16 passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_get_sub_tensor___bf16 failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_creation_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_creation_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_creation_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_tensor_creation_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_tensor_creation_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_tensor_creation_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_data_loading_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_data_loading_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_data_loading_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_tensor_data_loading_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_tensor_data_loading_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_tensor_data_loading_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_clone_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_clone_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_clone_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_tensor_clone_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_tensor_clone_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_tensor_clone_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_set_get_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_set_get_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_set_get_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_tensor_set_get_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_tensor_set_get_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_tensor_set_get_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_add_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_add_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_add_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_add_inplace_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_add_inplace_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_add_inplace_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_add_scalar_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_add_scalar_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_add_scalar_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_add_scalar_inplace_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_add_scalar_inplace_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_add_scalar_inplace_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_sub_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_sub_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_sub_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_sub_inplace_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_sub_inplace_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_sub_inplace_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_sub_scalar_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_sub_scalar_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_sub_scalar_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_sub_scalar_inplace_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_sub_scalar_inplace_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_sub_scalar_inplace_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_mul_ellement_wise_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_mul_ellement_wise_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_mul_ellement_wise_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_mul_ellement_wise_inplace_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_mul_ellement_wise_inplace_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_mul_ellement_wise_inplace_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_mul_scalar_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_mul_scalar_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_mul_scalar_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_mul_scalar_inplace_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_mul_scalar_inplace_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_mul_scalar_inplace_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_div_element_wise_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_div_element_wise_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_div_element_wise_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_div_element_wise_inplace_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_div_element_wise_inplace_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_div_element_wise_inplace_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_div_scalar_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_div_scalar_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_div_scalar_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_div_scalar_inplace_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_div_scalar_inplace_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_div_scalar_inplace_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_dot_product_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_dot_product_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_dot_product_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_cross_product_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_cross_product_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_cross_product_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_cross_product_inplace_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_cross_product_inplace_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_cross_product_inplace_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_norm1_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_norm1_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_norm1_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_norm2_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_norm2_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_norm2_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_norm3_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_norm3_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_norm3_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_normalization_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_normalization_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_normalization_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_normalization_inplace_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_normalization_inplace_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_normalization_inplace_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_get_sub_tensor_float(arena) == 0) {
         CCB_INFO("\e[32m[V] test_get_sub_tensor_float passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_get_sub_tensor_float failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_creation_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_creation_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_creation_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_tensor_creation_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_tensor_creation_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_tensor_creation_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_data_loading_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_data_loading_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_data_loading_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_tensor_data_loading_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_tensor_data_loading_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_tensor_data_loading_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_clone_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_clone_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_clone_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_tensor_clone_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_tensor_clone_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_tensor_clone_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_set_get_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_set_get_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_set_get_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_tensor_set_get_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_tensor_set_get_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_tensor_set_get_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_add_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_add_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_add_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_add_inplace_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_add_inplace_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_add_inplace_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_add_scalar_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_add_scalar_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_add_scalar_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_add_scalar_inplace_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_add_scalar_inplace_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_add_scalar_inplace_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_sub_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_sub_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_sub_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_sub_inplace_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_sub_inplace_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_sub_inplace_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_sub_scalar_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_sub_scalar_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_sub_scalar_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_sub_scalar_inplace_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_sub_scalar_inplace_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_sub_scalar_inplace_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_mul_ellement_wise_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_mul_ellement_wise_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_mul_ellement_wise_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_mul_ellement_wise_inplace_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_mul_ellement_wise_inplace_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_mul_ellement_wise_inplace_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_mul_scalar_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_mul_scalar_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_mul_scalar_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_mul_scalar_inplace_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_mul_scalar_inplace_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_mul_scalar_inplace_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_div_element_wise_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_div_element_wise_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_div_element_wise_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_div_element_wise_inplace_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_div_element_wise_inplace_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_div_element_wise_inplace_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_div_scalar_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_div_scalar_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_div_scalar_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_div_scalar_inplace_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_div_scalar_inplace_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_div_scalar_inplace_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_dot_product_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_dot_product_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_dot_product_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_cross_product_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_cross_product_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_cross_product_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_cross_product_inplace_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_cross_product_inplace_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_cross_product_inplace_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_norm1_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_norm1_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_norm1_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_norm2_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_norm2_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_norm2_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_norm3_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_norm3_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_norm3_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_normalization_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_normalization_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_normalization_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_vector_normalization_inplace_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_vector_normalization_inplace_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_vector_normalization_inplace_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

     if (test_get_sub_tensor_double(arena) == 0) {
         CCB_INFO("\e[32m[V] test_get_sub_tensor_double passed\e[0m");
         passed++;
     } else {
         CCB_INFO("\e[31m[X] test_get_sub_tensor_double failed\e[0m");
         failed++;
     }
     total++;
     ccb_arena_reset(arena);

    CCB_INFO("Tests completed: \e[32m%d passed\e[0m, \e[31m%d failed\e[0m, %d total", passed, failed, total);
    CCB_INFO("Success rate: \e[33m%.2f%\e[0m", (passed / (float)total) * 100.0f);

    ccb_arena_free(arena);
    if (failed > 0) {
        CCB_ERROR("Some tests failed");
        return -1;
    }

    CCB_INFO("All tests passed successfully");
    ccb_GetLogFile();
    return 0;
}
