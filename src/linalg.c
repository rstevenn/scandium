#include "data.h"
#include "linalg.h"
#include "const.h"
#include "../ccbase/logs/log.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>


// #########################
// General vector operations
// #########################

cl_vector* cl_for_each_vector_op(cl_vector* a, cl_vector* b, cl_value_t (*func)(cl_value_t, cl_value_t), ccb_arena* arena) {
    if (a->size != b->size) {
        CCB_ERROR("Vector size mismatch: %u vs %u", a->size, b->size);
        return NULL;
    }
    if (a->type != b->type) {
        CCB_ERROR("Vector type mismatch: %d vs %d", a->type, b->type);
        return NULL;
    }

    cl_vector* result = cl_create_vector(a->size, a->type, arena);
    CCB_NOTNULL(result, "Failed to create result vector");

    for (uint32_t i = 0; i < a->size; i++) {
        switch (a->type) {
            case CL_float16: {
                __bf16* a_data = (__bf16*)a->data;
                __bf16* b_data = (__bf16*)b->data;
                __bf16* r_data = (__bf16*)result->data;
                r_data[i] = func(to_cl_value((float)a_data[i], CL_float16),
                                 to_cl_value((float)b_data[i], CL_float16)).value.f16;
                break;
            }
            case CL_float32: {
                float* a_data = (float*)a->data;
                float* b_data = (float*)b->data;
                float* r_data = (float*)result->data;
                r_data[i] = func(to_cl_value(a_data[i], CL_float32),
                                 to_cl_value(b_data[i], CL_float32)).value.f32;
                break;
            }
            case CL_float64: {
                double* a_data = (double*)a->data;
                double* b_data = (double*)b->data;
                double* r_data = (double*)result->data;
                r_data[i] = func(to_cl_value(a_data[i], CL_float64),
                                 to_cl_value(b_data[i], CL_float64)).value.f64;
                break;
            }
            default:
                CCB_ERROR("Unsupported CL_TYPES value %d", a->type);
                return NULL;
        }
    }

    return result;
}


cl_vector* cl_for_each_vector_op_inplace(cl_vector* a, cl_vector* b, cl_value_t (*func)(cl_value_t, cl_value_t)) {
    if (a->size != b->size) {
        CCB_ERROR("Vector size mismatch: %u vs %u", a->size, b->size);
        return NULL;
    }
    if (a->type != b->type) {
        CCB_ERROR("Vector type mismatch: %d vs %d", a->type, b->type);
        return NULL;
    }

    for (uint32_t i = 0; i < a->size; i++) {
        switch (a->type) {
            case CL_float16: {
                __bf16* a_data = (__bf16*)a->data;
                __bf16* b_data = (__bf16*)b->data;
                a_data[i] = func((cl_value_t){.type=CL_float16, .value.f16=a_data[i]},
                                 (cl_value_t){.type=CL_float16, .value.f16=b_data[i]}).value.f16;
                break;
            }
            case CL_float32: {
                float* a_data = (float*)a->data;
                float* b_data = (float*)b->data;
                a_data[i] = func((cl_value_t){.type=CL_float32, .value.f32=a_data[i]},
                                 (cl_value_t){.type=CL_float32, .value.f32=b_data[i]}).value.f32;
                break;
            }
            case CL_float64: {
                double* a_data = (double*)a->data;
                double* b_data = (double*)b->data;
                a_data[i] = func((cl_value_t){.type=CL_float64, .value.f64=a_data[i]},
                                 (cl_value_t){.type=CL_float64, .value.f64=b_data[i]}).value.f64;
                break;
            }
            default:
                CCB_ERROR("Unsupported CL_TYPES value %d", a->type);
                return NULL;
        }
    }

    return a;
}


cl_vector* cl_for_each_vector_scalar_op(cl_vector* a, cl_value_t b, cl_value_t (*func)(cl_value_t, cl_value_t), ccb_arena* arena) {
    if (a->type != b.type) {
        CCB_ERROR("Vector and scalar type mismatch: %d vs %d", a->type, b.type);
        return NULL;
    }
    cl_vector* result = cl_create_vector(a->size, a->type, arena);
    CCB_NOTNULL(result, "Failed to create result vector");

    for (uint32_t i = 0; i < a->size; i++) {
        switch (a->type) {
            case CL_float16: {
                __bf16* a_data = (__bf16*)a->data;
                __bf16* r_data = (__bf16*)result->data;
                r_data[i] = func((cl_value_t){.type=CL_float16, .value.f16=a_data[i]}, b).value.f16;
                break;
            }
            case CL_float32: {
                float* a_data = (float*)a->data;
                float* r_data = (float*)result->data;
                r_data[i] = func((cl_value_t){.type=CL_float32, .value.f32=a_data[i]}, b).value.f32;
                break;
            }
            case CL_float64: {
                double* a_data = (double*)a->data;
                double* r_data = (double*)result->data;
                r_data[i] = func((cl_value_t){.type=CL_float64, .value.f64=a_data[i]}, b).value.f64;
                break;
            }
            default:
                CCB_ERROR("Unsupported CL_TYPES value %d", a->type);
                return NULL;
        }
    }

    return result;
}

cl_vector* cl_for_each_vector_scalar_op_inplace(cl_vector* a, cl_value_t b, cl_value_t (*func)(cl_value_t, cl_value_t)) {
    if (a->type != b.type) {
        CCB_ERROR("Vector and scalar type mismatch: %d vs %d", a->type, b.type);
        return NULL;
    }

    for (uint32_t i = 0; i < a->size; i++) {
        switch (a->type) {
            case CL_float16: {
                __bf16* a_data = (__bf16*)a->data;
                a_data[i] = func((cl_value_t){.type=CL_float16, .value.f16=a_data[i]}, b).value.f16;
                break;
            }
            case CL_float32: {
                float* a_data = (float*)a->data;
                a_data[i] = func((cl_value_t){.type=CL_float32, .value.f32=a_data[i]}, b).value.f32;
                break;
            }
            case CL_float64: {
                double* a_data = (double*)a->data;
                a_data[i] = func((cl_value_t){.type=CL_float64, .value.f64=a_data[i]}, b).value.f64;
                break;
            }
            default:
                CCB_ERROR("Unsupported CL_TYPES value %d", a->type);
                return NULL;
        }
    }

    return a;
}


cl_value_t cl_vector_reduce(cl_vector* a, cl_value_t (*func)(cl_value_t, cl_value_t), cl_value_t initial) {
    if (a->size == 0) {
        return initial;
    }

    cl_value_t result = initial;

    for (uint32_t i = 0; i < a->size; i++) {

        result = func(result, cl_get_vector_element(a, i));
    }
    return result;
}


cl_vector* cl_vector_map(cl_vector* a, cl_value_t (*func)(cl_value_t), ccb_arena* arena) {
    cl_vector* result = cl_create_vector(a->size, a->type, arena);
    CCB_NOTNULL(result, "Failed to create result vector");

    for (uint32_t i = 0; i < a->size; i++) {
        cl_set_vector_element(result, i, func(cl_get_vector_element(a, i)));
    }

    return result;
}


cl_vector* cl_vector_map_inplace(cl_vector* a, cl_value_t (*func)(cl_value_t)){
    for (uint32_t i = 0; i < a->size; i++) {
        cl_set_vector_element(a, i, func(cl_get_vector_element(a, i)));
    }

    return a;
}

cl_vector* cl_vector_map_args(cl_vector* a, cl_value_t (*func)(cl_value_t, void*), ccb_arena* arena, void* args) {
    cl_vector* result = cl_create_vector(a->size, a->type, arena);
    CCB_NOTNULL(result, "Failed to create result vector");

    for (uint32_t i = 0; i < a->size; i++) {
        cl_set_vector_element(result, i, func(cl_get_vector_element(a, i), args));
    }

    return result;
}

cl_vector* cl_vector_map_args_inplace(cl_vector* a, cl_value_t (*func)(cl_value_t, void*), void* args) {
    
    for (uint32_t i = 0; i < a->size; i++) {
        cl_set_vector_element(a, i, func(cl_get_vector_element(a, i), args));
    }

    return a;
}



// #################
// Scalar operations
// #################
cl_value_t cl_scalar_add(cl_value_t a, cl_value_t b) {
    if (a.type != b.type) {
        CCB_ERROR("Value type mismatch: %d vs %d", a.type, b.type);
        return (cl_value_t){0};
    }

    switch (a.type) {
        case CL_float16:
            return (cl_value_t){.type=CL_float16, .value.f16 = a.value.f16 + b.value.f16};
        case CL_float32:
            return (cl_value_t){.type=CL_float32, .value.f32 = a.value.f32 + b.value.f32};
        case CL_float64:
            return (cl_value_t){.type=CL_float64, .value.f64 = a.value.f64 + b.value.f64};
        default:
            CCB_ERROR("Unsupported CL_TYPES value %d", a.type);
            return (cl_value_t){0};
    }
}

cl_value_t cl_scalar_add_args(cl_value_t a, void* args) {
    CCB_NOTNULL(args, "b is NULL");
    return cl_scalar_add(a, *(cl_value_t*)args);
}


cl_value_t cl_scalar_sub(cl_value_t a, cl_value_t b) {
    if (a.type != b.type) {
        CCB_ERROR("Value type mismatch: %d vs %d", a.type, b.type);
        return (cl_value_t){0};
    }

    switch (a.type) {
        case CL_float16:
            return (cl_value_t){.type=CL_float16, .value.f16 = a.value.f16 - b.value.f16};
        case CL_float32:
            return (cl_value_t){.type=CL_float32, .value.f32 = a.value.f32 - b.value.f32};
        case CL_float64:
            return (cl_value_t){.type=CL_float64, .value.f64 = a.value.f64 - b.value.f64};
        default:
            CCB_ERROR("Unsupported CL_TYPES value %d", a.type);
            return (cl_value_t){0};
    }
}

cl_value_t cl_scalar_sub_args(cl_value_t a, void* args) {
    CCB_NOTNULL(args, "b is NULL");
    return cl_scalar_sub(a, *(cl_value_t*)args);
}


cl_value_t cl_scalar_mul(cl_value_t a, cl_value_t b) {
 if (a.type != b.type) {
        CCB_ERROR("Value type mismatch: %d vs %d", a.type, b.type);
        return (cl_value_t){0};
    }

    switch (a.type) {
        case CL_float16:
            return (cl_value_t){.type=CL_float16, .value.f16 = a.value.f16 * b.value.f16};
        case CL_float32:
            return (cl_value_t){.type=CL_float32, .value.f32 = a.value.f32 * b.value.f32};
        case CL_float64:
            return (cl_value_t){.type=CL_float64, .value.f64 = a.value.f64 * b.value.f64};
        default:
            CCB_ERROR("Unsupported CL_TYPES value %d", a.type);
            return (cl_value_t){0};
    }   
}

cl_value_t cl_scalar_mul_args(cl_value_t a, void* args) {
    CCB_NOTNULL(args, "b is NULL");
    return cl_scalar_mul(a, *(cl_value_t*)args);
}


cl_value_t cl_scalar_div(cl_value_t a, cl_value_t b) {
 if (a.type != b.type) {
        CCB_ERROR("Value type mismatch: %d vs %d", a.type, b.type);
        return (cl_value_t){0};
    }

    switch (a.type) {
        case CL_float16:
            return (cl_value_t){.type=CL_float16, .value.f16 = a.value.f16 / b.value.f16};
        case CL_float32:
            return (cl_value_t){.type=CL_float32, .value.f32 = a.value.f32 / b.value.f32};
        case CL_float64:
            return (cl_value_t){.type=CL_float64, .value.f64 = a.value.f64 / b.value.f64};
        default:
            CCB_ERROR("Unsupported CL_TYPES value %d", a.type);
            return (cl_value_t){0};
    }   
}

cl_value_t cl_scalar_div_args(cl_value_t a, void* args) {
    CCB_NOTNULL(args, "b is NULL");
    return cl_scalar_div(a, *(cl_value_t*)args);
}


cl_value_t cl_scalar_abs(cl_value_t a) {
    switch (a.type) {
        case CL_float16:
            return (cl_value_t){.type=CL_float16, .value.f16 = (float)fabsf((float)a.value.f16)};
        case CL_float32:
            return (cl_value_t){.type=CL_float32, .value.f32 = fabsf(a.value.f32)};
        case CL_float64:
            return (cl_value_t){.type=CL_float64, .value.f64 = fabs(a.value.f64)};
        default:
            CCB_ERROR("Unsupported CL_TYPES value %d", a.type);
            return (cl_value_t){0};
    }
}


cl_value_t cl_scalar_pow(cl_value_t a, cl_value_t b) {
    if (a.type != b.type) {
        CCB_ERROR("Value type mismatch: %d vs %d", a.type, b.type);
        return (cl_value_t){0};
    }

    switch (a.type) {
        case CL_float16:
            return (cl_value_t){.type=CL_float16, .value.f16 = powf((float)a.value.f16, (float)b.value.f16)};
        case CL_float32:
            return (cl_value_t){.type=CL_float32, .value.f32 = powf(a.value.f32, b.value.f32)};
        case CL_float64:
            return (cl_value_t){.type=CL_float64, .value.f64 = pow(a.value.f64, b.value.f64)};
        default:
            CCB_ERROR("Unsupported CL_TYPES value %d", a.type);
            return (cl_value_t){0};
    }
}

cl_value_t cl_scalar_pow_args(cl_value_t a, void* args) {
    CCB_NOTNULL(args, "b is NULL");
    return cl_scalar_pow(a, *(cl_value_t*)args);
}


cl_value_t cl_scalar_root(cl_value_t a, cl_value_t b) {
    if (a.type != b.type) {
        CCB_ERROR("Value type mismatch: %d vs %d", a.type, b.type);
        return (cl_value_t){0};
    }
    
    switch (a.type) {
        case CL_float16:
            return (cl_value_t){.type=CL_float16, .value.f16 = powf((float)a.value.f16, 1.0/(float)b.value.f16)};
        case CL_float32:
            return (cl_value_t){.type=CL_float32, .value.f32 = powf(a.value.f32, 1.0/b.value.f32)};
        case CL_float64:
            return (cl_value_t){.type=CL_float64, .value.f64 = pow(a.value.f64, 1.0/b.value.f64)};
        default:
            CCB_ERROR("Unsupported CL_TYPES value %d", a.type);
            return (cl_value_t){0};
    }
}

cl_value_t cl_scalar_root_args(cl_value_t a, void* args) {
    CCB_NOTNULL(args, "b is NULL");
    return cl_scalar_root(a, *(cl_value_t*)args);
}






// ##########################
// Specific vector operations
// ##########################

// add
cl_vector* cl_vector_add(cl_vector* a, cl_vector* b, ccb_arena* arena) {
    if (a->size != b->size) {
        CCB_ERROR("Vector size mismatch: %u vs %u", a->size, b->size);
        return NULL;
    }

    return cl_for_each_vector_op(a, b, cl_scalar_add, arena);
}


cl_vector* cl_vector_add_inplace(cl_vector* a, cl_vector* b) {
    if (a->size != b->size) {
        CCB_ERROR("Vector size mismatch: %u vs %u", a->size, b->size);
        return NULL;
    }

    return cl_for_each_vector_op_inplace(a, b, cl_scalar_add);
}

cl_vector* cl_vector_add_scalar(cl_vector* a, cl_value_t b, ccb_arena* arena) {
    return cl_for_each_vector_scalar_op(a, b, cl_scalar_add, arena);
}

cl_vector* cl_vector_add_scalar_inplace(cl_vector* a, cl_value_t b) {
    return cl_for_each_vector_scalar_op_inplace(a, b, cl_scalar_add);
}

// sub
cl_vector* cl_vector_sub(cl_vector* a, cl_vector* b, ccb_arena* arena) {
    if (a->size != b->size) {
        CCB_ERROR("Vector size mismatch: %u vs %u", a->size, b->size);
        return NULL;
    }

    return cl_for_each_vector_op(a, b, cl_scalar_sub, arena);
}

cl_vector* cl_vector_sub_inplace(cl_vector* a, cl_vector* b) {
    if (a->size != b->size) {
        CCB_ERROR("Vector size mismatch: %u vs %u", a->size, b->size);
        return NULL;
    }

    return cl_for_each_vector_op_inplace(a, b, cl_scalar_sub);
}

cl_vector* cl_vector_sub_scalar(cl_vector* a, cl_value_t b, ccb_arena* arena) {
    return cl_for_each_vector_scalar_op(a, b, cl_scalar_sub, arena);
}

cl_vector* cl_vector_sub_scalar_inplace(cl_vector* a, cl_value_t b) {
    return cl_for_each_vector_scalar_op_inplace(a, b, cl_scalar_sub);
}


// mult
cl_vector* cl_vector_mul_ellement_wise(cl_vector* a, cl_vector* b, ccb_arena* arena) {
    if (a->size != b->size) {
        CCB_ERROR("Vector size mismatch: %u vs %u", a->size, b->size);
        return NULL;
    }

    return cl_for_each_vector_op(a, b, cl_scalar_mul, arena);
}

cl_vector* cl_vector_mul_ellement_wise_inplace(cl_vector* a, cl_vector* b) {
    if (a->size != b->size) {
        CCB_ERROR("Vectro size mismatch: %u vs %u", a->size, b->size);
        return NULL;
    }

    return cl_for_each_vector_op_inplace(a, b, cl_scalar_mul);
}

cl_vector* cl_vector_mul_scalar(cl_vector* a, cl_value_t b, ccb_arena* arena) {
    return cl_for_each_vector_scalar_op(a, b, cl_scalar_mul, arena);
}

cl_vector* cl_vector_mul_scalar_inplace(cl_vector* a, cl_value_t b) {
    return cl_for_each_vector_scalar_op_inplace(a, b, cl_scalar_mul);
}

// div
cl_vector* cl_vector_div_ellement_wise(cl_vector* a, cl_vector* b, ccb_arena* arena) {
    if (a->size != b->size) {
        CCB_ERROR("Vector size mismatch: %u vs %u", a->size, b->size);
        return NULL;
    }

    return cl_for_each_vector_op(a, b, cl_scalar_div, arena);
}

cl_vector* cl_vector_div_ellement_wise_inplace(cl_vector* a, cl_vector* b) {
    if (a->size != b->size) {
       CCB_ERROR("Vector size mismatch: %u vs %u", a->size, b->size);
        return NULL;
    }

    return cl_for_each_vector_op_inplace(a, b, cl_scalar_div);
}

cl_vector* cl_vector_div_scalar(cl_vector* a, cl_value_t b, ccb_arena* arena) {
    return cl_for_each_vector_scalar_op(a, b, cl_scalar_div, arena);
}

cl_vector* cl_vector_div_scalar_inplace(cl_vector* a, cl_value_t b) {
    return cl_for_each_vector_scalar_op_inplace(a, b, cl_scalar_div);
}

// ##########################
// Advanced Vector operations
// ##########################

// dot
cl_value_t cl_vector_dot(cl_vector* a, cl_vector* b) {
    if (a->size != b->size) {
        CCB_ERROR("Vector size mismatch: %u vs %u", a->size, b->size);
        return (cl_value_t){0};
    }
    
    if (a->type != b->type) {
        CCB_ERROR("Vector type mismatch: %d vs %d", a->type, b->type);
        return (cl_value_t){0};
    }

    cl_value_t out;
    out.type = a->type;
    
    switch (a->type) {
        case CL_float16: {
            out.value.f16 = 0;
            __bf16* a_data = (__bf16*)a->data;
            __bf16* b_data = (__bf16*)b->data;
            for (uint32_t i = 0; i < a->size; i++) {
                out = cl_scalar_add(out, (cl_value_t){.type=CL_float16, .value.f16=a_data[i] * b_data[i]});
            }
            break;

        case CL_float32: {
            out.value.f32 = 0;
            float* a_data = (float*)a->data;
            float* b_data = (float*)b->data;
            for (uint32_t i = 0; i < a->size; i++) {
                out = cl_scalar_add(out, (cl_value_t){.type=CL_float32, .value.f32=a_data[i] * b_data[i]});
            }
            break;
        }

        case CL_float64: {
            out.value.f64 = 0;
            double* a_data = (double*)a->data;
            double* b_data = (double*)b->data;
            for (uint32_t i = 0; i < a->size; i++) {
                out = cl_scalar_add(out, (cl_value_t){.type=CL_float64, .value.f64=a_data[i] * b_data[i]});
            }
            break;
        }   

        default:
            CCB_ERROR("Unsupported CL_TYPES value %d", a->type);
            return (cl_value_t){0};
        }
    }
    return out;
}   

// cross
cl_vector* cl_vector_cross(cl_vector* a, cl_vector* b, ccb_arena* arena) {
    if (a->size != 3 || b->size != 3) {
        CCB_ERROR("Vector size mismatch: %u vs %u", a->size, b->size);
        return NULL;
    }

    if (a->type != b->type) {
        CCB_ERROR("Vector type mismatch: %d vs %d", a->type, b->type);
        return NULL;
    }

    cl_vector* out = cl_create_vector(a->size, a->type, arena);
    CCB_NOTNULL(out, "Failed to create result vector");

    switch (a->type) {
        case CL_float16: {
            __bf16* a_data = (__bf16*)a->data;
            __bf16* b_data = (__bf16*)b->data;
            __bf16* out_data = (__bf16*)out->data;
            out_data[0] = a_data[1] * b_data[2] - a_data[2] * b_data[1];
            out_data[1] = a_data[2] * b_data[0] - a_data[0] * b_data[2];
            out_data[2] = a_data[0] * b_data[1] - a_data[1] * b_data[0];
            break;
        }

        case CL_float32: {
            float* a_data = (float*)a->data;
            float* b_data = (float*)b->data;
            float* out_data = (float*)out->data;
            out_data[0] = a_data[1] * b_data[2] - a_data[2] * b_data[1];
            out_data[1] = a_data[2] * b_data[0] - a_data[0] * b_data[2];
            out_data[2] = a_data[0] * b_data[1] - a_data[1] * b_data[0];
            break;
        }

        case CL_float64: {
            double* a_data = (double*)a->data;
            double* b_data = (double*)b->data;
            double* out_data = (double*)out->data;
            out_data[0] = a_data[1] * b_data[2] - a_data[2] * b_data[1];
            out_data[1] = a_data[2] * b_data[0] - a_data[0] * b_data[2];
            out_data[2] = a_data[0] * b_data[1] - a_data[1] * b_data[0];
            break;
        }

        default: {
            CCB_ERROR("Unsupported CL_TYPES value %d", a->type);
            return NULL;
        }
    }
    return out;
}

cl_vector* cl_vector_cross_inplace(cl_vector* a, cl_vector* b) {
 if (a->size != 3 || b->size != 3) {
        CCB_ERROR("Vector size mismatch: %u vs %u", a->size, b->size);
        return NULL;
    }

    if (a->type != b->type) {
        CCB_ERROR("Vector type mismatch: %d vs %d", a->type, b->type);
        return NULL;
    }



    switch (a->type) {
        case CL_float16: {
            __bf16* a_data = (__bf16*)a->data;
            __bf16* b_data = (__bf16*)b->data;
            __bf16 a, b, c;
            a = a_data[1] * b_data[2] - a_data[2] * b_data[1];
            b = a_data[2] * b_data[0] - a_data[0] * b_data[2];
            c = a_data[0] * b_data[1] - a_data[1] * b_data[0];
            a_data[0] = a;
            a_data[1] = b;
            a_data[2] = c;
            break;
        }

        case CL_float32: {
            float* a_data = (float*)a->data;
            float* b_data = (float*)b->data;
            float a, b, c;
            a = a_data[1] * b_data[2] - a_data[2] * b_data[1];
            b = a_data[2] * b_data[0] - a_data[0] * b_data[2];
            c = a_data[0] * b_data[1] - a_data[1] * b_data[0];
            a_data[0] = a;
            a_data[1] = b;
            a_data[2] = c;
            break;
        }

        case CL_float64: {
            double* a_data = (double*)a->data;
            double* b_data = (double*)b->data;
            double a, b, c;
            a = a_data[1] * b_data[2] - a_data[2] * b_data[1];
            b = a_data[2] * b_data[0] - a_data[0] * b_data[2];
            c = a_data[0] * b_data[1] - a_data[1] * b_data[0];
            a_data[0] = a;
            a_data[1] = b;
            a_data[2] = c;
            break;
        }

        default: {
            CCB_ERROR("Unsupported CL_TYPES value %d", a->type);
            return NULL;
        }
    }
    return a;
}


cl_value_t norm_p_maper(cl_value_t a, void* b) {

    cl_value_t b_val = *(cl_value_t *)b;

    return cl_scalar_pow(a, b_val);
}


cl_value_t cl_vector_norm(cl_vector* a, uint32_t p, ccb_arena* tmp_arena) {
    if (a->size == 0) {
        CCB_ERROR("Vector size is 0");
        return (cl_value_t){0};
    }

    if (p == 0) {
        CCB_ERROR("p cannot be 0");
        return (cl_value_t){0};
    }


    cl_value_t out = to_cl_value(0.0, a->type);
    cl_vector_reduce(a, cl_scalar_add, out);


    if (p==1) {
        cl_vector* tmp =  cl_vector_map(a, cl_scalar_abs, tmp_arena);
        CCB_NOTNULL(tmp, "Failed to create temporary vector");
        return cl_vector_reduce(tmp, cl_scalar_add, out);

    } else {
        cl_value_t p_val = to_cl_value((float)p, a->type);
        cl_vector* tmp =  cl_vector_map_args(a, norm_p_maper, tmp_arena, &p_val);
        CCB_NOTNULL(tmp, "Failed to create temporary vector");
        return cl_scalar_root(cl_vector_reduce(tmp, cl_scalar_add, out), p_val);
    }
}


cl_vector* cl_vector_normalize(cl_vector* a, ccb_arena* arena) {
    cl_value_t norm = cl_vector_norm(a, 2, arena);
    if (cl_value_to_f64(norm) == 0) {
        CCB_ERROR("Vector norm is 0");
        return NULL;
    }

    return cl_vector_map_args(a, cl_scalar_div_args, arena, &norm);
}

cl_vector* cl_vector_normalize_inplace(cl_vector* a, ccb_arena* arena) {
    cl_value_t norm = cl_vector_norm(a, 2, arena);
    if (cl_value_to_f64(norm) == 0) {
        CCB_ERROR("Vector norm is 0");
        return NULL;
    }
    return cl_vector_map_args_inplace(a, cl_scalar_div_args, &norm);
}





