#include <stdio.h>

#include "data.h"

#define TEST_FILE "src/test.c"

typedef struct test_t {
    const char* data_type;
    const char* sc_type;
    const char* union_type;
    sc_TYPES sc_val;
} test_data;


test_data tests[] = {
    {"__bf16", "sc_float16", "f16", sc_float16},
    {"float", "sc_float32",  "f32", sc_float32},
    {"double", "sc_float64", "f64", sc_float64},
    
};



void helper_generate_test_run(FILE* file, const char* test_name, const char* data_type) {
    if (data_type == NULL) {
        fprintf(file, "     if (test_%s_(arena) == 0) {\n", test_name);
    } else {
        fprintf(file, "     if (test_%s_%s(arena) == 0) {\n", test_name, data_type);
    }
    
    fprintf(file, "         CCB_INFO(\"\\e[32m[V] test_%s_%s passed\\e[0m\");\n", test_name, data_type);
    fprintf(file, "         passed++;\n");
    fprintf(file, "     } else {\n");
    fprintf(file, "         CCB_INFO(\"\\e[31m[X] test_%s_%s failed\\e[0m\");\n", test_name, data_type);
    fprintf(file, "         failed++;\n");
    fprintf(file, "     }\n");
    fprintf(file, "     total++;\n");
    fprintf(file, "     ccb_arena_reset(arena);\n\n");
}


void gen_test_dims_creation(FILE* file, test_data test) {
    fprintf(file, "int test_dims_creation_(ccb_arena* arena){\n");
    fprintf(file, "    sc_dimensions* dims = sc_create_dimensions(2, arena, (uint32_t[]){2, 5});\n");
    fprintf(file, "    if (!dims) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create dimensions\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n");
    fprintf(file, "    for (uint32_t i = 0; i < 2; i++) {\n");
    fprintf(file, "        if (dims->dims[i] != (uint32_t[]){2, 5}[i]) {\n");
    fprintf(file, "            CCB_WARNING(\"Dimensions creation failed\");\n");
    fprintf(file, "            return -1;\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_indices_creation(FILE* file, test_data test) {
    fprintf(file, "int test_indices_creation_(ccb_arena* arena){\n");
    fprintf(file, "    sc_index* index = sc_create_index(2, arena, (uint32_t[]){1, 2});\n");
    fprintf(file, "    if (!index) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create index\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n");
    fprintf(file, "    for (uint32_t i = 0; i < 2; i++) {\n");
    fprintf(file, "        if (index->indices[i] != (uint32_t[]){1, 2}[i]) {\n");
    fprintf(file, "            CCB_WARNING(\"Index creation failed\");\n");
    fprintf(file, "            return -1;\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_slice_creation(FILE* file, test_data test) {
    fprintf(file, "int test_slice_creation_(ccb_arena* arena){\n");
    fprintf(file, "    sc_slice* slice = sc_create_slice(2, arena, (uint32_t[]){1, 2}, (uint32_t[]){3, 4});\n");
    fprintf(file, "    if (!slice) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create slice\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n");
    fprintf(file, "    for (uint32_t i = 0; i < 2; i++) {\n");
    fprintf(file, "        if (slice->slices[i].start != (uint32_t[]){1, 2}[i] || slice->slices[i].end != (uint32_t[]){3, 4}[i]) {\n");
    fprintf(file, "            CCB_WARNING(\"Slice creation failed\");\n");
    fprintf(file, "            return -1;\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_vector_creation(FILE* file, test_data test) {
    fprintf(file, "int test_vector_creation_%s(ccb_arena* arena){\n", test.data_type);
    fprintf(file, "    sc_vector* vector = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_tensor_creation(FILE* file, test_data test) {
    fprintf(file, "int test_tensor_creation_%s(ccb_arena* arena){\n", test.data_type);
    fprintf(file, "    sc_dimensions* dims = sc_create_dimensions(2, arena, (uint32_t[]){2, 5});\n");
    fprintf(file, "    if (!dims) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create dimensions\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_tensor* tensor = sc_create_tensor(dims, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!tensor) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create tensor\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_vector_data_loading(FILE* file, test_data test) {
    fprintf(file, "int test_vector_data_loading_%s(ccb_arena* arena){\n", test.data_type);
    fprintf(file, "    sc_vector* vector = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    %s data[10] = {0", test.data_type);
    for (int i = 1; i < 10; i++) {
        fprintf(file, ", %d", i);
    }
    fprintf(file, "};\n");
    fprintf(file, "    sc_data_to_vector(vector, data, 10);\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_tensor_data_loading(FILE* file, test_data test) {
    fprintf(file, "int test_tensor_data_loading_%s(ccb_arena* arena){\n", test.data_type);
    fprintf(file, "    sc_dimensions* dims = sc_create_dimensions(2, arena, (uint32_t[]){2, 5});\n");
    fprintf(file, "    if (!dims) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create dimensions\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_tensor* tensor = sc_create_tensor(dims, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!tensor) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create tensor\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    %s data[10] = {0", test.data_type);
    for (int i = 1; i < 10; i++) {
        fprintf(file, ", %d", i);
    }
    fprintf(file, "};\n");
    fprintf(file, "    sc_data_to_tensor(tensor, data, 10);\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_vector_clone(FILE* file, test_data test) {
    fprintf(file, "int test_vector_clone_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    sc_vector* vector = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = to_sc_value((%s)i, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector, i, val);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_vector* clone = sc_clone_vector(vector, arena);\n");
    fprintf(file, "    if (!clone) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to clone vector\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t original_val = sc_get_vector_element(vector, i);\n");
    fprintf(file, "        sc_value_t clone_val = sc_get_vector_element(clone, i);\n");
    fprintf(file, "        if (original_val.value.%s != clone_val.value.%s) {\n", test.union_type, test.union_type);
    fprintf(file, "            CCB_WARNING(\"Vector clone failed\");\n");
    fprintf(file, "            return -1;\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_tensor_clone(FILE* file, test_data test) {
    fprintf(file, "int test_tensor_clone_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    sc_dimensions* dims = sc_create_dimensions(2, arena, (uint32_t[]){3, 4});\n");
    fprintf(file, "    if (!dims) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create dimensions\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_tensor* tensor = sc_create_tensor(dims, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!tensor) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create tensor\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 3; i++) {\n");
    fprintf(file, "        for (uint32_t j = 0; j < 4; j++) {\n");
    fprintf(file, "            sc_index index;\n");
    fprintf(file, "            index.count = 2;\n");
    fprintf(file, "            uint32_t idxs[2] = {i, j};\n");
    fprintf(file, "            index.indices = idxs;\n");
    fprintf(file, "            sc_value_t val = to_sc_value((%s)(i * 4 + j), %s);\n", test.data_type, test.sc_type);
    fprintf(file, "            sc_set_tensor_element(tensor, &index, val);\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_tensor* clone = sc_clone_tensor(tensor, arena);\n");
    fprintf(file, "    if (!clone) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to clone tensor\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 3; i++) {\n");
    fprintf(file, "        for (uint32_t j = 0; j < 4; j++) {\n");
    fprintf(file, "            sc_index index;\n");
    fprintf(file, "            index.count = 2;\n");
    fprintf(file, "            uint32_t idxs[2] = {i, j};\n");
    fprintf(file, "            index.indices = idxs;\n\n");
    fprintf(file, "            sc_value_t original_val = sc_get_tensor_element(tensor, &index);\n");
    fprintf(file, "            sc_value_t clone_val = sc_get_tensor_element(clone, &index);\n");
    fprintf(file, "            if (original_val.value.%s != clone_val.value.%s) {\n", test.union_type, test.union_type);
    fprintf(file, "                CCB_WARNING(\"tensor clone failed\");\n");
    fprintf(file, "                return -1;\n");
    fprintf(file, "            }\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");

}


void gen_test_vector_set_get(FILE* file, test_data test) {
    fprintf(file, "int test_vector_set_get_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    sc_vector* vector = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = to_sc_value((%s)i, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector, i, val);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = sc_get_vector_element(vector, i);\n");
    fprintf(file, "        if (val.type != %s || val.value.%s != (%s)i) {\n", test.sc_type, test.union_type, test.data_type);
    fprintf(file, "            CCB_WARNING(\"Vector set/get mismatch at index %%u: expected %%f, got %%f\", i, (%s)i, val.value.%s);\n", test.data_type, test.union_type);
    fprintf(file, "            return -1;\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_tensor_set_get(FILE* file, test_data test) {
    fprintf(file, "int test_tensor_set_get_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    sc_dimensions* dims = sc_create_dimensions(2, arena, (uint32_t[]){2, 5});\n");
    fprintf(file, "    if (!dims) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create dimensions\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_tensor* tensor = sc_create_tensor(dims, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!tensor) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create tensor\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 2; i++) {\n");
    fprintf(file, "        for (uint32_t j = 0; j < 5; j++) {\n");
    fprintf(file, "            sc_index index;\n"); 
    fprintf(file, "            index.count = 2;\n");
    fprintf(file, "            uint32_t idxs[2] = {i, j};\n");
    fprintf(file, "            index.indices = idxs;\n");
    fprintf(file, "            sc_value_t val = to_sc_value((%s)(i * 5 + j) * 3.0%s, %s);\n", test.data_type, (test.data_type[0] == 'f' ? "f" : ""), test.sc_type);
    fprintf(file, "            sc_set_tensor_element(tensor, &index, val);\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 2; i++) {\n");
    fprintf(file, "        for (uint32_t j = 0; j < 5; j++) {\n");
    fprintf(file, "            sc_index index;\n");
    fprintf(file, "            index.count = 2;\n");
    fprintf(file, "            uint32_t idxs[2] = {i, j};\n");
    fprintf(file, "            index.indices = idxs;\n\n");
    fprintf(file, "            sc_value_t val = sc_get_tensor_element(tensor, &index);\n");
    fprintf(file, "            if (val.type != %s || val.value.%s != ((%s)(i * 5 + j) * 3.0%s)) {\n", test.sc_type, test.union_type, test.data_type, (test.data_type[0] == 'f' ? "f" : ""));
    fprintf(file, "                CCB_WARNING(\"tensor set/get mismatch at index [%%u, %%u]: expected %%f, got %%f\", i, j, (%s)(i * 5 + j) * 3.0%s, val.value.%s);\n", test.data_type, (test.data_type[0] == 'f' ? "f" : ""), test.union_type);
    fprintf(file, "                return -1;\n");
    fprintf(file, "            }\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}


void gen_test_vector_add(FILE* file, test_data test) {
    fprintf(file, "int test_vector_add_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    sc_vector* vector1 = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector1) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector1\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_vector* vector2 = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector2) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector2\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val1 = to_sc_value((%s)i, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector1, i, val1);\n");
    fprintf(file, "        sc_value_t val2 = to_sc_value((%s)(i * 2), %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector2, i, val2);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_vector* result = sc_vector_add(vector1, vector2, arena);\n");
    fprintf(file, "    if (!result) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create result vector\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = sc_get_vector_element(result, i);\n");
    fprintf(file, "        if (val.type != %s || val.value.%s != (%s)(i + i * 2)) {\n", test.sc_type, test.union_type, test.data_type);
    fprintf(file, "            CCB_WARNING(\"Vector addition mismatch at index %%u: expected %%f, got %%f\", i, (%s)(i + i * 2), val.value.%s);\n", test.data_type, test.union_type);
    fprintf(file, "            return -1;\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_vector_add_inplace(FILE* file, test_data test) {
    fprintf(file, "int test_vector_add_inplace_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    sc_vector* vector1 = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector1) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector1\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_vector* vector2 = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector2) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector2\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val1 = to_sc_value((%s)i, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector1, i, val1);\n");
    fprintf(file, "        sc_value_t val2 = to_sc_value((%s)(i * 2), %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector2, i, val2);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val1 = sc_get_vector_element(vector1, i);\n");
    fprintf(file, "        sc_value_t val2 = sc_get_vector_element(vector2, i);\n");
    fprintf(file, "        sc_value_t sum;\n");
    fprintf(file, "        sum.type = %s;\n", test.sc_type);
    if (test.sc_val == sc_float16) {
        fprintf(file, "        sum.value.f16 = (__bf16)((float)val1.value.f16 + (float)val2.value.f16);\n");
    } else if (test.sc_val == sc_float32) {
        fprintf(file, "        sum.value.f32 = val1.value.f32 + val2.value.f32;\n");
    } else if (test.sc_val == sc_float64) {
        fprintf(file, "        sum.value.f64 = val1.value.f64 + val2.value.f64;\n");
    } else {
        fprintf(file, "        // Unsupported type for addition\n");
        fprintf(file, "        return -1;\n");
    }
    fprintf(file, "        sc_set_vector_element(vector1, i, sum);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = sc_get_vector_element(vector1, i);\n");
    fprintf(file, "        if (val.type != %s || val.value.%s != (%    s)(i + i * 2)) {\n", test.sc_type, test.union_type, test.data_type);
    fprintf(file, "            CCB_WARNING(\"In-place vector addition mismatch at index %%u : expected %%f, got %%f\", i, (%s)(i + i * 2), val.value.%s);\n", test.data_type, test.union_type);
    fprintf(file, "            return -1;\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_vector_add_scalar(FILE* file, test_data test) {
    fprintf(file, "int test_vector_add_scalar_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    sc_vector* vector = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = to_sc_value((%s)i, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector, i, val);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_value_t scalar = to_sc_value((%s)5, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "    sc_vector* result = sc_vector_add_scalar(vector, scalar, arena);\n");
    fprintf(file, "    if (!result) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create result vector\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = sc_get_vector_element(result, i);\n");
    fprintf(file, "        if (val.type != %s || val.value.%s != (%s)(i + 5)) {\n", test.sc_type, test.union_type, test.data_type);
    fprintf(file, "            CCB_WARNING(\"Vector-scalar addition mismatch at index %%u   : expected %%f, got %%f\", i, (%s)(i + 5), val.value.%s);\n", test.data_type, test.union_type);
    fprintf(file, "            return -1;\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

int gen_test_vector_add_scalar_inplace(FILE* file, test_data test) {
    fprintf(file, "int test_vector_add_scalar_inplace_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    sc_vector* vector = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = to_sc_value((%s)i, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector, i, val);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_value_t scalar = to_sc_value((%s)5, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = sc_get_vector_element(vector, i);\n");
    fprintf(file, "        sc_value_t sum;\n");
    fprintf(file, "        sum.type = %s;\n", test.sc_type);
    fprintf(file, "        sum.value.%s = val.value.%s + scalar.value.%s;\n", test.union_type, test.union_type, test.union_type);
    fprintf(file, "        sc_set_vector_element(vector, i, sum);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = sc_get_vector_element(vector, i);\n");
    fprintf(file, "        if (val.type != %s || val.value.%s != (%s)(i + 5)) {\n", test.sc_type, test.union_type, test.data_type);
    fprintf(file, "            CCB_WARNING(\"In-place vector-scalar addition mismatch at index %%u : expected %%f, got %%f\", i, (%s)(i + 5), val.value.%s);\n", test.data_type, test.union_type);
    fprintf(file, "            return -1;\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_vector_sub(FILE* file, test_data test) {
    fprintf(file, "int test_vector_sub_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    sc_vector* vector1 = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector1) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector1\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_vector* vector2 = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector2) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector2\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val1 = to_sc_value((%s)(i * 3), %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector1, i, val1);\n");
    fprintf(file, "        sc_value_t val2 = to_sc_value((%s)(i * 2), %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector2, i, val2);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_vector* result = sc_vector_sub(vector1, vector2, arena);\n");
    fprintf(file, "    if (!result) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create result vector\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n"); 
    fprintf(file, "        sc_value_t val = sc_get_vector_element(result, i);\n");
    fprintf(file, "        if (val.type != %s || val.value.%s != (%s)(i * 3 - i * 2)) {\n", test.sc_type, test.union_type, test.data_type);
    fprintf(file, "            CCB_WARNING(\"Vector subtraction mismatch at index %%u: expected %%f, got %%f\", i, (%s)(i * 3 - i * 2), val.value.%s);\n", test.data_type, test.union_type);
    fprintf(file, "            return -1;\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_vector_sub_inplace(FILE* file, test_data test) {
    fprintf(file, "int test_vector_sub_inplace_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    sc_vector* vector1 = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector1) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector1\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_vector* vector2 = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector2) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector2\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val1 = to_sc_value((%s)(i * 3), %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector1, i, val1);\n");
    fprintf(file, "        sc_value_t val2 = to_sc_value((%s)(i * 2), %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector2, i, val2);\n");
    fprintf(file, "    }\n\n");    
    fprintf(file, "    vector1 = sc_vector_sub_inplace(vector1, vector2);\n");
    fprintf(file, "    if (!vector1) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create result vector\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = sc_get_vector_element(vector1, i);\n");
    fprintf(file, "        if (val.type != %s || val.value.%s != (%s)(i * 3 - i * 2)) {\n", test.sc_type, test.union_type, test.data_type);
    fprintf(file, "            CCB_WARNING(\"Vector subtraction mismatch at index %%u: expected %%f, got %%f\", i, (%s)(i * 3 - i * 2), val.value.%s);\n", test.data_type, test.union_type);
    fprintf(file, "            return -1;\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_vector_sub_scalar(FILE* file, test_data test) {
    fprintf(file, "int test_vector_sub_scalar_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    sc_vector* vector = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = to_sc_value((%s)(i * 3), %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector, i, val);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_value_t scalar = to_sc_value((%s)2, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "    sc_vector* result = sc_vector_sub_scalar(vector, scalar, arena);\n");
    fprintf(file, "    if (!result) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create result vector\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (int32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = sc_get_vector_element(result, i);\n");
    fprintf(file, "        if (val.type != %s || val.value.%s != (%s)(i * 3 - 2)) {\n", test.sc_type, test.union_type, test.data_type);
    fprintf(file, "            CCB_WARNING(\"Vector-scalar subtraction mismatch at index %%u: expected %%f, got %%f\", i, (%s)(i * 3 - 2), val.value.%s);\n", test.data_type, test.union_type);
    fprintf(file, "            return -1;\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_vector_sub_scalar_inplace(FILE* file, test_data test) {
    fprintf(file, "int test_vector_sub_scalar_inplace_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    sc_vector* vector = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = to_sc_value((%s)(i * 3), %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector, i, val);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_value_t scalar = to_sc_value((%s)2, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = sc_get_vector_element(vector, i);\n");
    fprintf(file, "        val.value.%s -= scalar.value.%s;\n", test.union_type, test.union_type);
    fprintf(file, "        sc_set_vector_element(vector, i, val);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (int32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "         sc_value_t val = sc_get_vector_element(vector, i);\n");
    fprintf(file, "         if (val.type != %s || val.value.%s != (%s)(i * 3 - 2)) {\n", test.sc_type, test.union_type, test.data_type, test.data_type);
    fprintf(file, "             CCB_WARNING(\"Vector-scalar subtraction mismatch at index %%u: expected %%f, got %%f\", i, (%s)(i * 3 - 2), val.value.%s);\n", test.data_type, test.union_type);
    fprintf(file, "             return -1;\n");
    fprintf(file, "         }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}


void gen_test_vector_mul_ellement_wise(FILE* file, test_data test) {
    fprintf(file, "int test_vector_mul_ellement_wise_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    sc_vector* vector1 = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector1) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector1\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_vector* vector2 = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector2) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector2\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val1 = to_sc_value((%s)i, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector1, i, val1);\n");
    fprintf(file, "        sc_value_t val2 = to_sc_value((%s)(i * 2), %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector2, i, val2);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_vector* result = sc_vector_mul_ellement_wise(vector1, vector2, arena);\n");
    fprintf(file, "    if (!result) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create result vectore\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = sc_get_vector_element(result, i);\n");
    fprintf(file, "        if (val.type != %s || val.value.%s != (%s)(i * i * 2)) {\n", test.sc_type, test.union_type, test.data_type);
    fprintf(file, "            CCB_WARNING(\"Vector element-wise multiplication mismatch at index %%u: expected %%f, got %%f\", i, (%s)(i * i * 2), val.value.%s);\n", test.data_type, test.union_type);
    fprintf(file, "            return -1;\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_vector_mul_ellement_wise_inplace(FILE* file, test_data test) {
    fprintf(file, "int test_vector_mul_ellement_wise_inplace_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    sc_vector* vector1 = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector1) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector1\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_vector* vector2 = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector2) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector2\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val1 = to_sc_value((%s)i, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector1, i, val1);\n");
    fprintf(file, "        sc_value_t val2 = to_sc_value((%s)(i * 2), %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector2, i, val2);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_vector* result = sc_vector_mul_ellement_wise_inplace(vector1, vector2);\n");
    fprintf(file, "    if (!result) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create result vectore\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = sc_get_vector_element(result, i);\n");
    fprintf(file, "        if (val.type != %s || val.value.%s != (%s)(i * i * 2)) {\n", test.sc_type, test.union_type, test.data_type);
    fprintf(file, "            CCB_WARNING(\"Vector element-wise multiplication mismatch at index %%u: expected %%f, got %%f\", i, (%s)(i * i * 2), val.value.%s);\n", test.data_type, test.union_type);
    fprintf(file, "            return -1;\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_vector_mul_scalar(FILE* file, test_data test) {
    fprintf(file, "int test_vector_mul_scalar_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    sc_vector* vector = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = to_sc_value((%s)i, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector, i, val);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_value_t scalar = to_sc_value((%s)2, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "    sc_vector* result = sc_vector_mul_scalar(vector, scalar, arena);\n");
    fprintf(file, "    if (!result){\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create result vector\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = sc_get_vector_element(result, i);\n");
    fprintf(file, "        if (val.type != %s || val.value.%s != (%s)(2 * i)) {\n", test.sc_type, test.union_type, test.data_type);
    fprintf(file, "            CCB_WARNING(\"Vector-scalar multiplication mismatch at index %%u: expected %%f, got %%f\", i, (%s)(2 * i), val.value.%s);\n", test.data_type, test.union_type);
    fprintf(file, "            return -1;\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_vector_mul_scalar_inplace(FILE* file, test_data test) {
    fprintf(file, "int test_vector_mul_scalar_inplace_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    sc_vector* vector = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = to_sc_value((%s)i, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector, i, val);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_value_t scalar = to_sc_value((%s)2, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "    sc_vector* result = sc_vector_mul_scalar_inplace(vector, scalar);\n");
    fprintf(file, "    if (!result){\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create result vector\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = sc_get_vector_element(result, i);\n");
    fprintf(file, "        if (val.type != %s || val.value.%s != (%s)(2 * i)) {\n", test.sc_type, test.union_type, test.data_type);
    fprintf(file, "            CCB_WARNING(\"Vector-scalar multiplication mismatch at index %%u: expected %%f, got %%f\", i, (%s)(2 * i), val.value.%s);\n", test.data_type, test.union_type);
    fprintf(file, "            return -1;\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_vector_div_element_wise(FILE* file, test_data test) {
    fprintf(file, "int test_vector_div_element_wise_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    sc_vector* vector1 = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector1) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector1\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_vector* vector2 = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector2) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector2\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val1 = to_sc_value((%s)i, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector1, i, val1);\n");
    fprintf(file, "        sc_value_t val2 = to_sc_value((%s)(i+1), %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector2, i, val2);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_vector* result = sc_vector_div_ellement_wise(vector1, vector2, arena);\n");
    fprintf(file, "    if (!result) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create result vectore\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = sc_get_vector_element(result, i);\n");
    fprintf(file, "        %s expected = (%s)i / (%s)(i +1);\n", test.data_type, test.data_type, test.data_type);
    fprintf(file, "        if (val.type != %s || val.value.%s != expected) {\n", test.sc_type, test.union_type, test.data_type  );
    fprintf(file, "            CCB_WARNING(\"Vector element-wise multiplication mismatch at index %%u: expected %%f, got %%f\", i, expected, (float)val.value.%s);\n", test.union_type);
    fprintf(file, "            return -1;\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_vector_div_element_wise_inplace(FILE* file, test_data test) {
    fprintf(file, "int test_vector_div_element_wise_inplace_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    sc_vector* vector1 = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector1) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector1\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_vector* vector2 = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector2) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector2\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val1 = to_sc_value((%s)i, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector1, i, val1);\n");
    fprintf(file, "        sc_value_t val2 = to_sc_value((%s)(i+1), %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector2, i, val2);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_vector* result = sc_vector_div_ellement_wise_inplace(vector1, vector2);\n");
    fprintf(file, "    if (!result) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create result vectore\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = sc_get_vector_element(result, i);\n");
    fprintf(file, "        %s expected = (%s)i / (%s)(i +1);\n", test.data_type, test.data_type, test.data_type);
    fprintf(file, "        if (val.type != %s || val.value.%s != expected) {\n", test.sc_type, test.union_type, test.data_type  );
    fprintf(file, "            CCB_WARNING(\"Vector element-wise multiplication mismatch at index %%u: expected %%f, got %%f\", i, expected, (float)val.value.%s);\n", test.union_type);
    fprintf(file, "            return -1;\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_vector_div_scalar(FILE* file, test_data test) {
    fprintf(file, "int test_vector_div_scalar_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    sc_vector* vector1 = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector1) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector1\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = to_sc_value((%s)i, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector1, i, val);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_value_t scalar = to_sc_value((%s)2, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "    sc_vector* result = sc_vector_div_scalar(vector1, scalar, arena);\n");
    fprintf(file, "    if (!result) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create result vector\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = sc_get_vector_element(result, i);\n");
    fprintf(file, "        %s expected = (%s)i / (%s)2;\n", test.data_type, test.data_type, test.data_type);
    fprintf(file, "        if (val.type != %s || val.value.%s != expected) {\n", test.sc_type, test.union_type, test.data_type  );
    fprintf(file, "            CCB_WARNING(\"Vector-scalar division mismatch at index %%u: expected %%f, got %%f\", i, expected, (float)val.value.%s);\n", test.union_type);
    fprintf(file, "            return -1;\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}
    
void gen_test_vector_div_scalar_inplace(FILE* file, test_data test) {
    fprintf(file, "int test_vector_div_scalar_inplace_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    sc_vector* vector1 = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector1) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector1\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = to_sc_value((%s)i, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector1, i, val);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_value_t scalar = to_sc_value((%s)2, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "    sc_vector* result = sc_vector_div_scalar_inplace(vector1, scalar);\n");
    fprintf(file, "    if (!result) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create result vector\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = sc_get_vector_element(result, i);\n");
    fprintf(file, "        %s expected = (%s)i / (%s)2;\n", test.data_type, test.data_type, test.data_type);
    fprintf(file, "        if (val.type != %s || val.value.%s != expected) {\n", test.sc_type, test.union_type, test.data_type  );
    fprintf(file, "            CCB_WARNING(\"Vector-scalar division mismatch at index %%u: expected %%f, got %%f\", i, expected, (float)val.value.%s);\n", test.union_type);
    fprintf(file, "            return -1;\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_vector_dot_product(FILE* file, test_data test) {
    fprintf(file, "int test_vector_dot_product_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    sc_vector* vector1 = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector1) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector1\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_vector* vector2 = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector2) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector1\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val1 = to_sc_value((%s)i/2, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector1, i, val1);\n");
    fprintf(file, "        sc_value_t val2 = to_sc_value((%s)(i * 2), %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector2, i, val2);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_value_t result = sc_vector_dot(vector1, vector2);\n\n");
    fprintf(file, "    %s expected = 0;\n", test.data_type);
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        expected += (%s)i/2 * (%s)(i * 2);\n", test.data_type, test.data_type);
    fprintf(file, "    }\n\n");
    fprintf(file, "    if (result.type != %s || result.value.%s != expected) {\n", test.sc_type, test.union_type);
    fprintf(file, "        CCB_WARNING(\"Vector dot product mismatch: expected %%f, got %%f\", (float)expected, (float)result.value.%s);\n", test.union_type);
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_vector_cross_product(FILE* file, test_data test) {
    fprintf(file, "int test_vector_cross_product_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    sc_vector* vector1 = sc_create_vector(3, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector1) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector1\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_vector* vector2 = sc_create_vector(3, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector2){\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector1\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 3; i++) {\n");
    fprintf(file, "        sc_value_t val1 = to_sc_value((%s)i, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector1, i, val1);\n");
    fprintf(file, "        sc_value_t val2 = to_sc_value((%s)(i + 1), %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector2, i, val2);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_vector* result = sc_vector_cross(vector1, vector2, arena);\n");
    fprintf(file, "    if (!result) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create result vector\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_vector* expected = sc_create_vector(3, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!expected) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create expected vector\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    %s* a = (%s*)vector1->data;\n", test.data_type, test.data_type);
    fprintf(file, "    %s* b = (%s*)vector2->data;\n", test.data_type, test.data_type);
    fprintf(file, "    %s* c = (%s*)expected->data;\n", test.data_type, test.data_type);
    fprintf(file, "    c[0] = a[1] * b[2] - a[2] * b[1];\n");
    fprintf(file, "    c[1] = a[2] * b[0] - a[0] * b[2];\n");
    fprintf(file, "    c[2] = a[0] * b[1] - a[1] * b[0];\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 3; i++) {\n");
    fprintf(file, "        sc_value_t val = sc_get_vector_element(result, i);\n");
    fprintf(file, "        sc_value_t expected_val = sc_get_vector_element(expected, i);\n");
    fprintf(file, "        if (val.type != %s || val.value.%s != expected_val.value.%s) {\n", test.sc_type, test.union_type, test.union_type);
    fprintf(file, "            CCB_WARNING(\"Vector dot product mismatch: expected %%f, got %%f\", (float)expected_val.value.%s, (float)val.value.%s);\n", test.union_type, test.union_type);
    fprintf(file, "            return -1;\n");
    fprintf(file, "         }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_vector_cross_product_inplace(FILE* file, test_data test) {
    fprintf(file, "int test_vector_cross_product_inplace_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    sc_vector* vector1 = sc_create_vector(3, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector1) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector1\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_vector* vector2 = sc_create_vector(3, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!vector2){\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create vector1\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 3; i++) {\n");
    fprintf(file, "        sc_value_t val1 = to_sc_value((%s)i, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector1, i, val1);\n");
    fprintf(file, "        sc_value_t val2 = to_sc_value((%s)(i + 1), %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector2, i, val2);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_vector* expected = sc_create_vector(3, %s, arena);\n", test.sc_type);
    fprintf(file, "    if (!expected) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create expected vector\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    %s* a = (%s*)vector1->data;\n", test.data_type, test.data_type);
    fprintf(file, "    %s* b = (%s*)vector2->data;\n", test.data_type, test.data_type);
    fprintf(file, "    %s* c = (%s*)expected->data;\n", test.data_type, test.data_type);
    fprintf(file, "    c[0] = a[1] * b[2] - a[2] * b[1];\n");
    fprintf(file, "    c[1] = a[2] * b[0] - a[0] * b[2];\n");
    fprintf(file, "    c[2] = a[0] * b[1] - a[1] * b[0];\n\n");
    fprintf(file, "    sc_vector* result = sc_vector_cross_inplace(vector1, vector2);\n");
    fprintf(file, "    if (!result) {\n");
    fprintf(file, "        CCB_WARNING(\"Failed to create result vector\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 3; i++) {\n");
    fprintf(file, "        sc_value_t val = sc_get_vector_element(result, i);\n");
    fprintf(file, "        sc_value_t expected_val = sc_get_vector_element(expected, i);\n");
    fprintf(file, "        if (val.type != %s || val.value.%s != expected_val.value.%s) {\n", test.sc_type, test.union_type, test.union_type);
    fprintf(file, "            CCB_WARNING(\"Vector dot product mismatch: expected %%f, got %%f\", (float)expected_val.value.%s, (float)val.value.%s);\n", test.union_type, test.union_type);
    fprintf(file, "            return -1;\n");
    fprintf(file, "         }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_vector_norm1(FILE* file, test_data test) {
    fprintf(file, "int test_vector_norm1_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    sc_vector* vector1 = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    CCB_NOTNULL(vector1, \"Failed to create vector1\");\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = to_sc_value((%s)i, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector1, i, val);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_value_t result = sc_vector_norm(vector1, 1, arena);\n");
    fprintf(file, "    if (result.type != %s) {\n", test.sc_type);
    fprintf(file, "        CCB_WARNING(\"Vector norm mismatch: expected %%f, got %%f\", (float)0, (float)result.value.%s);\n", test.union_type);
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_value_t expected = to_sc_value((%s)0, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        expected = sc_scalar_add(expected, sc_scalar_abs(sc_get_vector_element(vector1, i)));\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    if (result.value.%s != expected.value.%s) {\n", test.union_type, test.union_type);
    fprintf(file, "         CCB_WARNING(\"Vector norm mismatch: expected %%f, got %%f\", (float)expected.value.%s, (float)result.value.%s);\n", test.union_type, test.union_type);
    fprintf(file, "         return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_vector_norm2(FILE* file, test_data test) {
    fprintf(file, "int test_vector_norm2_%s(ccb_arena* arena) {\n", test.data_type);
   fprintf(file, "    sc_vector* vector1 = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    CCB_NOTNULL(vector1, \"Failed to create vector1\");\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = to_sc_value((%s)i, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector1, i, val);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_value_t result = sc_vector_norm(vector1, 2, arena);\n");
    fprintf(file, "    if (result.type != %s) {\n", test.sc_type);
    fprintf(file, "        CCB_WARNING(\"Vector norm mismatch: expected %%f, got %%f\", (float)0, (float)result.value.%s);\n", test.union_type);
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_value_t expected = to_sc_value((%s)0, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "    sc_value_t val_p = to_sc_value((%s)2, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        expected = sc_scalar_add(expected, sc_scalar_pow(sc_get_vector_element(vector1, i), val_p));\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    expected = sc_scalar_root(expected, val_p);\n");
    fprintf(file, "    if (result.value.%s != expected.value.%s) {\n", test.union_type, test.union_type);
    fprintf(file, "         CCB_WARNING(\"Vector norm mismatch: expected %%f, got %%f\", (float)expected.value.%s, (float)result.value.%s);\n", test.union_type, test.union_type);
    fprintf(file, "         return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}    

void gen_test_vector_norm3(FILE* file, test_data test) {
    fprintf(file, "int test_vector_norm3_%s(ccb_arena* arena) {\n", test.data_type);
   fprintf(file, "    sc_vector* vector1 = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    CCB_NOTNULL(vector1, \"Failed to create vector1\");\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = to_sc_value((%s)i, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector1, i, val);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_value_t result = sc_vector_norm(vector1, 3, arena);\n");
    fprintf(file, "    if (result.type != %s) {\n", test.sc_type);
    fprintf(file, "        CCB_WARNING(\"Vector norm mismatch: expected %%f, got %%f\", (float)0, (float)result.value.%s);\n", test.union_type);
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_value_t expected = to_sc_value((%s)0, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "    sc_value_t val_p = to_sc_value((%s)3, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        expected = sc_scalar_add(expected, sc_scalar_pow(sc_get_vector_element(vector1, i), val_p));\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    expected = sc_scalar_root(expected, val_p);\n");
    fprintf(file, "    if (result.value.%s != expected.value.%s) {\n", test.union_type, test.union_type);
    fprintf(file, "         CCB_WARNING(\"Vector norm mismatch: expected %%f, got %%f\", (float)expected.value.%s, (float)result.value.%s);\n", test.union_type, test.union_type);
    fprintf(file, "         return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_vector_normalization(FILE* file, test_data test) {
    fprintf(file, "int test_vector_normalization_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    sc_vector* vector1 = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    CCB_NOTNULL(vector1, \"Failed to create vector1\");\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = to_sc_value((%s)i, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector1, i, val);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_vector* result = sc_vector_normalize(vector1, arena);\n");
    fprintf(file, "    if (!result) {\n");
    fprintf(file, "         CCB_WARNING(\"Failed to create result vector\");\n");
    fprintf(file, "         return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_value_t norm = sc_vector_norm(vector1, 2, arena);\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = sc_get_vector_element(result, i);\n");
    fprintf(file, "        sc_value_t expected = sc_scalar_div(sc_get_vector_element(vector1, i), norm);\n");
    fprintf(file, "        if (val.type != %s || val.value.%s != expected.value.%s) {\n", test.sc_type, test.union_type, test.union_type);
    fprintf(file, "            CCB_WARNING(\"Vector normalization mismatch at index %%u: expected %%f, got %%f\", i, (float)expected.value.%s, (float)val.value.%s);\n", test.union_type, test.union_type);
    fprintf(file, "            return -1;\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");    
}

void gen_test_vector_normalization_inplace(FILE* file, test_data test) {
    fprintf(file, "int test_vector_normalization_inplace_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    sc_vector* vector1 = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    CCB_NOTNULL(vector1, \"Failed to create vector1\");\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = to_sc_value((%s)i, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector1, i, val);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    sc_value_t norm = sc_vector_norm(vector1, 2, arena);\n");
    fprintf(file, "    sc_vector* result = sc_vector_normalize_inplace(vector1, arena);\n");
    fprintf(file, "    if (!result) {\n");
    fprintf(file, "         CCB_WARNING(\"Failed to create result vector\");\n");
    fprintf(file, "         return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = sc_get_vector_element(result, i);\n");
    fprintf(file, "        sc_value_t expected = sc_scalar_div(to_sc_value((%s)i, %s), norm);\n", test.data_type, test.sc_type);
    fprintf(file, "        if (val.type != %s || val.value.%s != expected.value.%s) {\n", test.sc_type, test.union_type, test.union_type);
    fprintf(file, "            CCB_WARNING(\"Vector normalization mismatch at index %%u: expected %%f, got %%f\", i, (float)expected.value.%s, (float)val.value.%s);\n", test.union_type, test.union_type);    
    fprintf(file, "            return -1;\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");    
}


void gen_test_get_sub_tensor(FILE* file, test_data test) {
    fprintf(file, "int test_get_sub_tensor_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    uint32_t shape[3] = {4, 4, 4};\n");
    fprintf(file, "    sc_dimensions* dims = sc_create_dimensions(3, arena, shape);\n");
    fprintf(file, "    sc_tensor* tensor = sc_create_tensor(dims, %s, arena);\n", test.sc_type);
    fprintf(file, "    CCB_NOTNULL(tensor, \"Failed to create tensor\");\n\n");
    fprintf(file, "    sc_index* zero_idx = sc_create_index(3, arena, (uint32_t[]){0, 0, 0});\n");
    fprintf(file, "    CCB_NOTNULL(zero_idx, \"Failed to create zero index\");\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 64; i++) {\n");
    fprintf(file, "        sc_value_t val = to_sc_value((%s)i, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        for (uint32_t j = 0; j < 3; j++) {\n");
    fprintf(file, "            zero_idx->indices[j] = (i / (uint32_t)pow(4, 2 - j)) %% 4;\n");
    fprintf(file, "        }\n");
    fprintf(file, "        sc_set_tensor_element(tensor, zero_idx, val);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    uint32_t start[3] = {1, 0};\n");
    fprintf(file, "    sc_index* indices = sc_create_index(2, arena, start);\n");
    fprintf(file, "    CCB_NOTNULL(indices, \"Failed to create indices\");\n\n");
    fprintf(file, "    sc_tensor* sub_tensor = sc_get_sub_tensor(tensor, indices, arena);\n");
    fprintf(file, "    CCB_NOTNULL(sub_tensor, \"Failed to create sub-tensor\");\n\n");
    fprintf(file, "    uint32_t expected_shape[1] = {4};\n");
    fprintf(file, "    if (sub_tensor->dims->dims_count != 1 || memcmp(sub_tensor->dims->dims, expected_shape, sizeof(expected_shape)) != 0) {\n");
    fprintf(file, "        CCB_WARNING(\"Sub-tensor shape mismatch: expected [4], got [%%u, %%u, %%u]\", sub_tensor->dims->dims[0], sub_tensor->dims->dims[1], sub_tensor->dims->dims[2]);\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 4; i++) {\n");
    fprintf(file, "        sc_index* idx = sc_create_index(1, arena, &i);\n");
    fprintf(file, "        sc_value_t val = sc_get_tensor_element(sub_tensor, idx);\n");
    fprintf(file, "        %s expected = (%s)(16 + i);\n", test.data_type, test.data_type);
    fprintf(file, "        if (val.type != %s || val.value.%s != expected) {\n", test.sc_type, test.union_type);
    fprintf(file, "            CCB_WARNING(\"Sub-tensor element mismatch at index %%u: expected %%f, got %%f\", i, (float)expected, (float)val.value.%s);\n", test.union_type);
    fprintf(file, "            return -1;\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}

void gen_test_get_slice_vector(FILE* file, test_data test) {
    fprintf(file, "int test_get_slice_vector_%s(ccb_arena* arena) {\n   ", test.data_type);
    fprintf(file, "    sc_vector* vector = sc_create_vector(10, %s, arena);\n", test.sc_type);
    fprintf(file, "    CCB_NOTNULL(vector, \"Failed to create vector\");\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 10; i++) {\n");
    fprintf(file, "        sc_value_t val = to_sc_value((%s)i, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        sc_set_vector_element(vector, i, val);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    uint32_t start = 2;\n");
    fprintf(file, "    uint32_t end = 7;\n");
    fprintf(file, "    sc_slice* slice = sc_create_slice(1, arena, &start, &end);\n");
    fprintf(file, "    CCB_NOTNULL(slice, \"Failed to create slice\");\n\n");
    fprintf(file, "    sc_vector* sub_vector = sc_get_vector_slice(vector, slice, arena);\n");
    fprintf(file, "    CCB_NOTNULL(sub_vector, \"Failed to create sub-vector\");\n\n");
    fprintf(file, "    if (sub_vector->size != 5) {\n");
    fprintf(file, "        CCB_WARNING(\"Sub-vector length mismatch: expected 5, got %%u\", sub_vector->size);\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 5; i++) {\n");
    fprintf(file, "        sc_value_t val = sc_get_vector_element(sub_vector, i);\n");
    fprintf(file, "        sc_value_t expected = to_sc_value((%s)(i + 2), %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        if (val.type != %s || val.value.%s != expected.value.%s) {\n", test.sc_type, test.union_type, test.union_type);
    fprintf(file, "            CCB_WARNING(\"Sub-vector element mismatch at index %%u: expected %%f, got %%f\", i, (float)expected.value.%s, (float)val.value.%s);\n", test.union_type, test.union_type);
    fprintf(file, "            return -1;\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}


void gen_test_get_slice_tensor(FILE* file, test_data test) {
    fprintf(file, "int test_get_slice_tensor_%s(ccb_arena* arena) {\n", test.data_type);
    fprintf(file, "    uint32_t shape[3] = {4, 4, 4};\n");
    fprintf(file, "    sc_dimensions* dims = sc_create_dimensions(3, arena, shape);\n");
    fprintf(file, "    sc_tensor* tensor = sc_create_tensor(dims, %s, arena);\n", test.sc_type);
    fprintf(file, "    CCB_NOTNULL(tensor, \"Failed to create tensor\");\n\n");
    fprintf(file, "    sc_index* zero_idx = sc_create_index(3, arena, (uint32_t[]){0, 0, 0});\n");
    fprintf(file, "    CCB_NOTNULL(zero_idx, \"Failed to create zero index\");\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 64; i++) {\n");
    fprintf(file, "        sc_value_t val = to_sc_value((%s)i, %s);\n", test.data_type, test.sc_type);
    fprintf(file, "        for (uint32_t j = 0; j < 3; j++) {\n");
    fprintf(file, "            zero_idx->indices[j] = (i / (uint32_t)pow(4, 2 - j)) %% 4;\n");
    fprintf(file, "        }\n");
    fprintf(file, "        sc_set_tensor_element(tensor, zero_idx, val);\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    uint32_t start[3] = {1, 1, 0};\n");
    fprintf(file, "    uint32_t end[3] = {3, 3, 4};\n");
    fprintf(file, "    sc_slice* slice = sc_create_slice(3, arena, start, end);\n");
    fprintf(file, "    CCB_NOTNULL(slice, \"Failed to create slice\");\n\n");
    fprintf(file, "    sc_tensor* sub_tensor = sc_get_tensor_slice(tensor, slice, arena);\n");
    fprintf(file, "    CCB_NOTNULL(sub_tensor, \"Failed to create sub-tensor\");\n\n");
    fprintf(file, "    uint32_t expected_shape[3] = {2, 2, 4};\n");
    fprintf(file, "    if (sub_tensor->dims->dims_count != 3 || memcmp(sub_tensor->dims->dims, expected_shape, sizeof(expected_shape)) != 0) {\n");
    fprintf(file, "        CCB_WARNING(\"Sub-tensor shape mismatch: expected [2, 2, 4], got [%%u, %%u, %%u]\", sub_tensor->dims->dims[0], sub_tensor->dims->dims[1], sub_tensor->dims->dims[2]);\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    for (uint32_t i = 0; i < 2; i++) {\n");
    fprintf(file, "        for (uint32_t j = 0; j < 2; j++) {\n");
    fprintf(file, "            for (uint32_t k = 0; k < 4; k++) {\n");
    fprintf(file, "                uint32_t idx_arr[3] = {i, j, k};\n");
    fprintf(file, "                sc_index* idx = sc_create_index(3, arena, idx_arr);\n");
    fprintf(file, "                sc_value_t val = sc_get_tensor_element(sub_tensor, idx);\n");
    fprintf(file, "                %s expected = (%s)(20 + k + 4*j + 16*i);\n", test.data_type, test.data_type);
    fprintf(file, "                if (val.type != %s || val.value.%s != expected) {\n", test.sc_type, test.union_type);
    fprintf(file, "                    CCB_WARNING(\"Sub-tensor element mismatch at index [%%u, %%u, %%u]: expected %%f, got %%f\", i, j, k, (float)expected, (float)val.value.%s);\n", test.union_type);
    fprintf(file, "                    return -1;\n");
    fprintf(file, "                }\n");
    fprintf(file, "            }\n");
    fprintf(file, "        }\n");
    fprintf(file, "    }\n\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n\n");
}



int main(void) {
    FILE* file = fopen(TEST_FILE, "w");

    if (!file) {
        fprintf(stderr, "Failed to open test file for writing\n");
        return -1;
    }
    
    // header
    fprintf(file, "#include \"scandium.h\"\n");
    fprintf(file, "#include <string.h>\n");
    fprintf(file, "#include <math.h>\n\n");



    // tests
    test_data dummy = {0};
    gen_test_dims_creation(file, dummy);
    gen_test_indices_creation(file, dummy);
    gen_test_slice_creation(file, dummy);

    
    for (int i = 0; i < sizeof(tests) / sizeof(test_data); i++) {
        gen_test_vector_creation(file, tests[i]);
        gen_test_tensor_creation(file, tests[i]);
        gen_test_vector_data_loading(file, tests[i]);
        gen_test_tensor_data_loading(file, tests[i]);
        gen_test_vector_clone(file, tests[i]);
        gen_test_tensor_clone(file, tests[i]);
        gen_test_vector_set_get(file, tests[i]);
        gen_test_tensor_set_get(file, tests[i]);
        gen_test_vector_add(file, tests[i]);
        gen_test_vector_add_inplace(file, tests[i]);
        gen_test_vector_add_scalar(file, tests[i]);
        gen_test_vector_add_scalar_inplace(file, tests[i]);
        gen_test_vector_sub(file, tests[i]);
        gen_test_vector_sub_inplace(file, tests[i]);
        gen_test_vector_sub_scalar(file, tests[i]);
        gen_test_vector_sub_scalar_inplace(file, tests[i]);
        gen_test_vector_mul_ellement_wise(file, tests[i]);
        gen_test_vector_mul_ellement_wise_inplace(file, tests[i]);
        gen_test_vector_mul_scalar(file, tests[i]);
        gen_test_vector_mul_scalar_inplace(file, tests[i]);
        gen_test_vector_div_element_wise(file, tests[i]);
        gen_test_vector_div_element_wise_inplace(file, tests[i]);
        gen_test_vector_div_scalar(file, tests[i]);
        gen_test_vector_div_scalar_inplace(file, tests[i]);
        gen_test_vector_dot_product(file, tests[i]);
        gen_test_vector_cross_product(file, tests[i]);
        gen_test_vector_cross_product_inplace(file, tests[i]);
        gen_test_vector_norm1(file, tests[i]);
        gen_test_vector_norm2(file, tests[i]);
        gen_test_vector_norm3(file, tests[i]);
        gen_test_vector_normalization(file, tests[i]);
        gen_test_vector_normalization_inplace(file, tests[i]);
        gen_test_get_sub_tensor(file, tests[i]);
        gen_test_get_slice_vector(file, tests[i]);
        gen_test_get_slice_tensor(file, tests[i]);
    }




    // main function
    fprintf(file, "int main(void) {\n");
    fprintf(file, "     ccb_arena* arena = ccb_init_arena();\n");
    fprintf(file, "     CCB_NOTNULL(arena, \"Failed to initialize memory arena\");\n\n");

    fprintf(file, "     int passed = 0;\n");
    fprintf(file, "     int failed = 0;\n");
    fprintf(file, "     int total = 0;\n\n");
    fprintf(file, "     ccb_InitLog(\"log/test.log\");\n\n");
    


    helper_generate_test_run(file, "dims_creation", dummy.data_type);
    helper_generate_test_run(file, "indices_creation", dummy.data_type);
    helper_generate_test_run(file, "slice_creation", dummy.data_type);


    for (int i = 0; i < sizeof(tests) / sizeof(test_data); i++) {
        helper_generate_test_run(file, "vector_creation", tests[i].data_type);
        helper_generate_test_run(file, "tensor_creation", tests[i].data_type);
        helper_generate_test_run(file, "vector_data_loading", tests[i].data_type);
        helper_generate_test_run(file, "tensor_data_loading", tests[i].data_type);
        helper_generate_test_run(file, "vector_clone", tests[i].data_type);
        helper_generate_test_run(file, "tensor_clone", tests[i].data_type);
        helper_generate_test_run(file, "vector_set_get", tests[i].data_type);
        helper_generate_test_run(file, "tensor_set_get", tests[i].data_type);
        helper_generate_test_run(file, "vector_add", tests[i].data_type);
        helper_generate_test_run(file, "vector_add_inplace", tests[i].data_type);
        helper_generate_test_run(file, "vector_add_scalar", tests[i].data_type);
        helper_generate_test_run(file, "vector_add_scalar_inplace", tests[i].data_type);
        helper_generate_test_run(file, "vector_sub", tests[i].data_type);
        helper_generate_test_run(file, "vector_sub_inplace", tests[i].data_type);
        helper_generate_test_run(file, "vector_sub_scalar", tests[i].data_type);
        helper_generate_test_run(file, "vector_sub_scalar_inplace", tests[i].data_type);
        helper_generate_test_run(file, "vector_mul_ellement_wise", tests[i].data_type);
        helper_generate_test_run(file, "vector_mul_ellement_wise_inplace", tests[i].data_type);
        helper_generate_test_run(file, "vector_mul_scalar", tests[i].data_type);
        helper_generate_test_run(file, "vector_mul_scalar_inplace", tests[i].data_type);
        helper_generate_test_run(file, "vector_div_element_wise", tests[i].data_type);
        helper_generate_test_run(file, "vector_div_element_wise_inplace", tests[i].data_type);
        helper_generate_test_run(file, "vector_div_scalar", tests[i].data_type);
        helper_generate_test_run(file, "vector_div_scalar_inplace", tests[i].data_type);
        helper_generate_test_run(file, "vector_dot_product", tests[i].data_type);
        helper_generate_test_run(file, "vector_cross_product", tests[i].data_type);
        helper_generate_test_run(file, "vector_cross_product_inplace", tests[i].data_type);
        helper_generate_test_run(file, "vector_norm1", tests[i].data_type);
        helper_generate_test_run(file, "vector_norm2", tests[i].data_type);
        helper_generate_test_run(file, "vector_norm3", tests[i].data_type);
        helper_generate_test_run(file, "vector_normalization", tests[i].data_type);
        helper_generate_test_run(file, "vector_normalization_inplace", tests[i].data_type);
        helper_generate_test_run(file, "get_sub_tensor", tests[i].data_type);
        helper_generate_test_run(file, "get_slice_vector", tests[i].data_type);
        helper_generate_test_run(file, "get_slice_tensor", tests[i].data_type);
    
    }


    fprintf(file, "    CCB_INFO(\"Tests completed: \\e[32m%%d passed\\e[0m, \\e[31m%%d failed\\e[0m, %%d total\", passed, failed, total);\n");
    fprintf(file, "    CCB_INFO(\"Success rate: \\e[33m%%.2f%%%%\\e[0m\", (passed / (float)total) * 100.0f);\n\n");
    fprintf(file, "    ccb_arena_free(arena);\n");
    fprintf(file, "    if (failed > 0) {\n");
    fprintf(file, "        CCB_ERROR(\"Some tests failed\");\n");
    fprintf(file, "        return -1;\n");
    fprintf(file, "    }\n\n"); 
    fprintf(file, "    CCB_INFO(\"All tests passed successfully\");\n");
    fprintf(file, "    ccb_GetLogFile();\n");
    fprintf(file, "    return 0;\n");
    fprintf(file, "}\n");


    fclose(file);
    return 0;
}