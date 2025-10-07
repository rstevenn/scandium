#include "data.h"
#include "linalg.h"
#include "sc_engine.h"
#include "const.h"
#include "ccbase/logs/log.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>


// #########################
// General vector operations
// #########################
ccb_arena* local_arena = NULL;

void init_tmp_arena() {
    if (local_arena == NULL) {
        local_arena = ccb_init_arena();
    }
}



sc_vector* sc_for_each_vector_op(sc_vector* a, sc_vector* b, sc_value_t (*func)(sc_value_t, sc_value_t), ccb_arena* arena) {
    if (a->size != b->size) {
        CCB_ERROR("Vector size mismatch: %u vs %u", a->size, b->size);
        return NULL;
    }
    if (a->type != b->type) {
        CCB_ERROR("Vector type mismatch: %d vs %d", a->type, b->type);
        return NULL;
    }

    sc_vector* result = sc_create_vector(a->size, a->type, arena);
    sc_task_result out;
    sc_task* task = sc_create_vector_element_wise_task(a, b, result, func, a->size, arena);
    
    sc_execute_task(task, sc_auto, &out, arena);

    if (!out.succes) {
        CCB_ERROR("Failed to execute vector operation task");
        return NULL;
    }
    
    return result;
}


sc_vector* sc_for_each_vector_op_inplace(sc_vector* a, sc_vector* b, sc_value_t (*func)(sc_value_t, sc_value_t)) {
    init_tmp_arena();

    if (a->size != b->size) {
        CCB_ERROR("Vector size mismatch: %u vs %u", a->size, b->size);
        return NULL;
    }
    
    if (a->type != b->type) {
        CCB_ERROR("Vector type mismatch: %d vs %d", a->type, b->type);
        return NULL;
    }

    sc_vector* result = a;
    sc_task_result out;
    sc_task* task = sc_create_vector_element_wise_task(a, b, result, func, a->size, local_arena);

    
    sc_execute_task(task, sc_auto, &out, local_arena);
    ccb_arena_reset(local_arena);

    if (!out.succes) {
        CCB_ERROR("Failed to execute vector operation task");
        return NULL;
    }
    
    return result;
}


sc_vector* sc_for_each_vector_scalar_op(sc_vector* a, sc_value_t b, sc_value_t (*func)(sc_value_t, sc_value_t), ccb_arena* arena) {
    if (a->type != b.type) {
        CCB_ERROR("Vector and scalar type mismatch: %d vs %d", a->type, b.type);
        return NULL;
    }
    sc_vector* result = sc_create_vector(a->size, a->type, arena);
    CCB_NOTNULL(result, "Failed to create result vector");

    sc_task* task = sc_create_vector_scalar_task(a, b, result, func, a->size, arena);
    
    sc_task_result out;
    sc_execute_task(task, sc_auto, &out, arena);

    if (!out.succes) {
        CCB_ERROR("Failed to execute vector operation task");
        return NULL;
    }
    
    return result;
}

sc_vector* sc_for_each_vector_scalar_op_inplace(sc_vector* a, sc_value_t b, sc_value_t (*func)(sc_value_t, sc_value_t)) {
    init_tmp_arena();

    if (a->type != b.type) {
        CCB_ERROR("Vector and scalar type mismatch: %d vs %d", a->type, b.type);
        return NULL;
    }
    sc_vector* result = a;
    CCB_NOTNULL(result, "Failed to create result vector");

    sc_task* task = sc_create_vector_scalar_task(a, b, result, func, a->size, local_arena);
    
    sc_task_result out;
    sc_execute_task(task, sc_auto, &out, local_arena);
    ccb_arena_reset(local_arena);

    

    if (!out.succes) {
        CCB_ERROR("Failed to execute vector operation task");
        return NULL;
    }
    
    return result;
}


sc_value_t sc_vector_reduce(sc_vector* a, sc_value_t (*func)(sc_value_t, sc_value_t), sc_value_t initial) {
    init_tmp_arena();

    if (a->size == 0) {
        return initial;
    }


    sc_task* task = sc_create_vector_reduce_task(a, initial, func, a->size, local_arena);

    sc_task_result out;
    sc_execute_task(task, sc_auto, &out, local_arena);
    ccb_arena_reset(local_arena);

    if (!out.succes) {
        CCB_ERROR("Failed to execute vector reduce task");
        return initial;
    }

    return out.scalar_result;
}


sc_vector* sc_vector_map(sc_vector* a, sc_value_t (*func)(sc_value_t), ccb_arena* arena) {
    sc_vector* result = sc_create_vector(a->size, a->type, arena);
    CCB_NOTNULL(result, "Failed to create result vector");

    sc_task* task = sc_create_vector_map_task(a, result, func, a->size, arena);

    sc_task_result out;
    sc_execute_task(task, sc_auto, &out, arena);
    
    if (!out.succes) {
        CCB_ERROR("Failed to execute vector map task");
        return NULL;
    }

    return result;

}


sc_vector* sc_vector_map_inplace(sc_vector* a, sc_value_t (*func)(sc_value_t)){
    init_tmp_arena();

    sc_vector* result = a;
    sc_task* task = sc_create_vector_map_task(a, result, func, a->size, local_arena);

    sc_task_result out;
    sc_execute_task(task, sc_auto, &out, local_arena);
    ccb_arena_reset(local_arena);
    
    if (!out.succes) {
        CCB_ERROR("Failed to execute vector map task");
        return NULL;
    }

    return result;
}


sc_vector* sc_vector_map_args(sc_vector* a, sc_value_t (*func)(sc_value_t, void*), ccb_arena* arena, void* args) {
    sc_vector* result = sc_create_vector(a->size, a->type, arena);
    CCB_NOTNULL(result, "Failed to create result vector");

    sc_task* task = sc_create_vector_map_args_task(a, result, func, args, a->size, arena);

    sc_task_result out;
    sc_execute_task(task, sc_auto, &out, arena);
    
    if (!out.succes) {
        CCB_ERROR("Failed to execute vector map args task");
        return NULL;
    }

    return result;
}

sc_vector* sc_vector_map_args_inplace(sc_vector* a, sc_value_t (*func)(sc_value_t, void*), void* args) {
    init_tmp_arena();

    sc_vector* result = a;
    sc_task* task = sc_create_vector_map_args_task(a, result, func, args, a->size, local_arena);

    sc_task_result out;
    sc_execute_task(task, sc_auto, &out, local_arena);
    ccb_arena_reset(local_arena);
    
    if (!out.succes) {
        CCB_ERROR("Failed to execute vector map args task");
        return NULL;
    }

    return result;
}



// #################
// Scalar operations
// #################
sc_value_t sc_scalar_add(sc_value_t a, sc_value_t b) {
    if (a.type != b.type) {
        CCB_ERROR("Value type mismatch: %d vs %d", a.type, b.type);
        return (sc_value_t){0};
    }

    switch (a.type) {
        case sc_float16:
            return (sc_value_t){.type=sc_float16, .value.f16 = a.value.f16 + b.value.f16};
        case sc_float32:
            return (sc_value_t){.type=sc_float32, .value.f32 = a.value.f32 + b.value.f32};
        case sc_float64:
            return (sc_value_t){.type=sc_float64, .value.f64 = a.value.f64 + b.value.f64};
        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", a.type);
            return (sc_value_t){0};
    }
}

sc_value_t sc_scalar_add_args(sc_value_t a, void* args) {
    CCB_NOTNULL(args, "b is NULL");
    return sc_scalar_add(a, *(sc_value_t*)args);
}


sc_value_t sc_scalar_sub(sc_value_t a, sc_value_t b) {
    if (a.type != b.type) {
        CCB_ERROR("Value type mismatch: %d vs %d", a.type, b.type);
        return (sc_value_t){0};
    }

    switch (a.type) {
        case sc_float16:
            return (sc_value_t){.type=sc_float16, .value.f16 = a.value.f16 - b.value.f16};
        case sc_float32:
            return (sc_value_t){.type=sc_float32, .value.f32 = a.value.f32 - b.value.f32};
        case sc_float64:
            return (sc_value_t){.type=sc_float64, .value.f64 = a.value.f64 - b.value.f64};
        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", a.type);
            return (sc_value_t){0};
    }
}

sc_value_t sc_scalar_sub_args(sc_value_t a, void* args) {
    CCB_NOTNULL(args, "b is NULL");
    return sc_scalar_sub(a, *(sc_value_t*)args);
}


sc_value_t sc_scalar_mul(sc_value_t a, sc_value_t b) {
 if (a.type != b.type) {
        CCB_ERROR("Value type mismatch: %d vs %d", a.type, b.type);
        return (sc_value_t){0};
    }

    switch (a.type) {
        case sc_float16:
            return (sc_value_t){.type=sc_float16, .value.f16 = a.value.f16 * b.value.f16};
        case sc_float32:
            return (sc_value_t){.type=sc_float32, .value.f32 = a.value.f32 * b.value.f32};
        case sc_float64:
            return (sc_value_t){.type=sc_float64, .value.f64 = a.value.f64 * b.value.f64};
        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", a.type);
            return (sc_value_t){0};
    }   
}

sc_value_t sc_scalar_mul_args(sc_value_t a, void* args) {
    CCB_NOTNULL(args, "b is NULL");
    return sc_scalar_mul(a, *(sc_value_t*)args);
}


sc_value_t sc_scalar_div(sc_value_t a, sc_value_t b) {
 if (a.type != b.type) {
        CCB_ERROR("Value type mismatch: %d vs %d", a.type, b.type);
        return (sc_value_t){0};
    }

    switch (a.type) {
        case sc_float16:
            return (sc_value_t){.type=sc_float16, .value.f16 = a.value.f16 / b.value.f16};
        case sc_float32:
            return (sc_value_t){.type=sc_float32, .value.f32 = a.value.f32 / b.value.f32};
        case sc_float64:
            return (sc_value_t){.type=sc_float64, .value.f64 = a.value.f64 / b.value.f64};
        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", a.type);
            return (sc_value_t){0};
    }   
}

sc_value_t sc_scalar_div_args(sc_value_t a, void* args) {
    CCB_NOTNULL(args, "b is NULL");
    return sc_scalar_div(a, *(sc_value_t*)args);
}


sc_value_t sc_scalar_abs(sc_value_t a) {
    switch (a.type) {
        case sc_float16:
            return (sc_value_t){.type=sc_float16, .value.f16 = (float)fabsf((float)a.value.f16)};
        case sc_float32:
            return (sc_value_t){.type=sc_float32, .value.f32 = fabsf(a.value.f32)};
        case sc_float64:
            return (sc_value_t){.type=sc_float64, .value.f64 = fabs(a.value.f64)};
        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", a.type);
            return (sc_value_t){0};
    }
}


sc_value_t sc_scalar_pow(sc_value_t a, sc_value_t b) {
    if (a.type != b.type) {
        CCB_ERROR("Value type mismatch: %d vs %d", a.type, b.type);
        return (sc_value_t){0};
    }

    switch (a.type) {
        case sc_float16:
            return (sc_value_t){.type=sc_float16, .value.f16 = powf((float)a.value.f16, (float)b.value.f16)};
        case sc_float32:
            return (sc_value_t){.type=sc_float32, .value.f32 = powf(a.value.f32, b.value.f32)};
        case sc_float64:
            return (sc_value_t){.type=sc_float64, .value.f64 = pow(a.value.f64, b.value.f64)};
        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", a.type);
            return (sc_value_t){0};
    }
}

sc_value_t sc_scalar_pow_args(sc_value_t a, void* args) {
    CCB_NOTNULL(args, "b is NULL");
    return sc_scalar_pow(a, *(sc_value_t*)args);
}


sc_value_t sc_scalar_root(sc_value_t a, sc_value_t b) {
    if (a.type != b.type) {
        CCB_ERROR("Value type mismatch: %d vs %d", a.type, b.type);
        return (sc_value_t){0};
    }
    
    switch (a.type) {
        case sc_float16:
            return (sc_value_t){.type=sc_float16, .value.f16 = powf((float)a.value.f16, 1.0/(float)b.value.f16)};
        case sc_float32:
            return (sc_value_t){.type=sc_float32, .value.f32 = powf(a.value.f32, 1.0/b.value.f32)};
        case sc_float64:
            return (sc_value_t){.type=sc_float64, .value.f64 = pow(a.value.f64, 1.0/b.value.f64)};
        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", a.type);
            return (sc_value_t){0};
    }
}

sc_value_t sc_scalar_root_args(sc_value_t a, void* args) {
    CCB_NOTNULL(args, "b is NULL");
    return sc_scalar_root(a, *(sc_value_t*)args);
}






// ##########################
// Specific vector operations
// ##########################

// add
sc_vector* sc_vector_add(sc_vector* a, sc_vector* b, ccb_arena* arena) {
    if (a->size != b->size) {
        CCB_ERROR("Vector size mismatch: %u vs %u", a->size, b->size);
        return NULL;
    }

    sc_vector* result = sc_for_each_vector_op(a, b, sc_scalar_add, arena);
    
    if (result == NULL) {
        CCB_ERROR("Failed to add vectors");
        return NULL;
    }

    return result;
}


sc_vector* sc_vector_add_inplace(sc_vector* a, sc_vector* b) {
    if (a->size != b->size) {
        CCB_ERROR("Vector size mismatch: %u vs %u", a->size, b->size);
        return NULL;
    }


    
    sc_vector* result = sc_for_each_vector_op_inplace(a, b, sc_scalar_add);
    if (result == NULL) {
        CCB_ERROR("Failed to add vectors");
        return NULL;
    }

    return result;
}

sc_vector* sc_vector_add_scalar(sc_vector* a, sc_value_t b, ccb_arena* arena) {
    return sc_for_each_vector_scalar_op(a, b, sc_scalar_add, arena);
}

sc_vector* sc_vector_add_scalar_inplace(sc_vector* a, sc_value_t b) {
    return sc_for_each_vector_scalar_op_inplace(a, b, sc_scalar_add);
}

// sub
sc_vector* sc_vector_sub(sc_vector* a, sc_vector* b, ccb_arena* arena) {
    if (a->size != b->size) {
        CCB_ERROR("Vector size mismatch: %u vs %u", a->size, b->size);
        return NULL;
    }

    
    sc_vector* result = sc_for_each_vector_op(a, b, sc_scalar_sub, arena);
    if (result == NULL) {
        CCB_ERROR("Failed to add vectors");
        return NULL;
    }

    return result;
}

sc_vector* sc_vector_sub_inplace(sc_vector* a, sc_vector* b) {
    if (a->size != b->size) {
        CCB_ERROR("Vector size mismatch: %u vs %u", a->size, b->size);
        return NULL;
    }

    return sc_for_each_vector_op_inplace(a, b, sc_scalar_sub);
}

sc_vector* sc_vector_sub_scalar(sc_vector* a, sc_value_t b, ccb_arena* arena) {
    return sc_for_each_vector_scalar_op(a, b, sc_scalar_sub, arena);
}

sc_vector* sc_vector_sub_scalar_inplace(sc_vector* a, sc_value_t b) {
    return sc_for_each_vector_scalar_op_inplace(a, b, sc_scalar_sub);
}


// mult
sc_vector* sc_vector_mul_ellement_wise(sc_vector* a, sc_vector* b, ccb_arena* arena) {
    if (a->size != b->size) {
        CCB_ERROR("Vector size mismatch: %u vs %u", a->size, b->size);
        return NULL;
    }

    return sc_for_each_vector_op(a, b, sc_scalar_mul, arena);
}

sc_vector* sc_vector_mul_ellement_wise_inplace(sc_vector* a, sc_vector* b) {
    if (a->size != b->size) {
        CCB_ERROR("Vectro size mismatch: %u vs %u", a->size, b->size);
        return NULL;
    }

    return sc_for_each_vector_op_inplace(a, b, sc_scalar_mul);
}

sc_vector* sc_vector_mul_scalar(sc_vector* a, sc_value_t b, ccb_arena* arena) {
    return sc_for_each_vector_scalar_op(a, b, sc_scalar_mul, arena);
}

sc_vector* sc_vector_mul_scalar_inplace(sc_vector* a, sc_value_t b) {
    return sc_for_each_vector_scalar_op_inplace(a, b, sc_scalar_mul);
}

// div
sc_vector* sc_vector_div_ellement_wise(sc_vector* a, sc_vector* b, ccb_arena* arena) {
    if (a->size != b->size) {
        CCB_ERROR("Vector size mismatch: %u vs %u", a->size, b->size);
        return NULL;
    }

    return sc_for_each_vector_op(a, b, sc_scalar_div, arena);
}

sc_vector* sc_vector_div_ellement_wise_inplace(sc_vector* a, sc_vector* b) {
    if (a->size != b->size) {
       CCB_ERROR("Vector size mismatch: %u vs %u", a->size, b->size);
        return NULL;
    }

    return sc_for_each_vector_op_inplace(a, b, sc_scalar_div);
}

sc_vector* sc_vector_div_scalar(sc_vector* a, sc_value_t b, ccb_arena* arena) {
    return sc_for_each_vector_scalar_op(a, b, sc_scalar_div, arena);
}

sc_vector* sc_vector_div_scalar_inplace(sc_vector* a, sc_value_t b) {
    return sc_for_each_vector_scalar_op_inplace(a, b, sc_scalar_div);
}

// ##########################
// Advanced Vector operations
// ##########################

// dot
sc_value_t sc_vector_dot(sc_vector* a, sc_vector* b) {
    init_tmp_arena();

    if (a->size != b->size) {
        CCB_ERROR("Vector size mismatch: %u vs %u", a->size, b->size);
        return (sc_value_t){0};
    }
    
    if (a->type != b->type) {
        CCB_ERROR("Vector type mismatch: %d vs %d", a->type, b->type);
        return (sc_value_t){0};
    }

    sc_vector* tmp = sc_vector_mul_ellement_wise(a, b, local_arena);

    sc_value_t out = sc_vector_reduce(tmp, sc_scalar_add, (sc_value_t){.type=a->type, .value.f32=0});
    ccb_arena_reset(local_arena);
    return out;
}   

// cross
sc_vector* sc_vector_cross(sc_vector* a, sc_vector* b, ccb_arena* arena) {
    if (a->size != 3 || b->size != 3) {
        CCB_ERROR("Vector size mismatch: %u vs %u", a->size, b->size);
        return NULL;
    }

    if (a->type != b->type) {
        CCB_ERROR("Vector type mismatch: %d vs %d", a->type, b->type);
        return NULL;
    }

    sc_vector* out = sc_create_vector(a->size, a->type, arena);
    CCB_NOTNULL(out, "Failed to create result vector");

    switch (a->type) {
        case sc_float16: {
            __bf16* a_data = (__bf16*)a->data;
            __bf16* b_data = (__bf16*)b->data;
            __bf16* out_data = (__bf16*)out->data;
            out_data[0] = a_data[1] * b_data[2] - a_data[2] * b_data[1];
            out_data[1] = a_data[2] * b_data[0] - a_data[0] * b_data[2];
            out_data[2] = a_data[0] * b_data[1] - a_data[1] * b_data[0];
            break;
        }

        case sc_float32: {
            float* a_data = (float*)a->data;
            float* b_data = (float*)b->data;
            float* out_data = (float*)out->data;
            out_data[0] = a_data[1] * b_data[2] - a_data[2] * b_data[1];
            out_data[1] = a_data[2] * b_data[0] - a_data[0] * b_data[2];
            out_data[2] = a_data[0] * b_data[1] - a_data[1] * b_data[0];
            break;
        }

        case sc_float64: {
            double* a_data = (double*)a->data;
            double* b_data = (double*)b->data;
            double* out_data = (double*)out->data;
            out_data[0] = a_data[1] * b_data[2] - a_data[2] * b_data[1];
            out_data[1] = a_data[2] * b_data[0] - a_data[0] * b_data[2];
            out_data[2] = a_data[0] * b_data[1] - a_data[1] * b_data[0];
            break;
        }

        default: {
            CCB_ERROR("Unsupported sc_TYPES value %d", a->type);
            return NULL;
        }
    }
    return out;
}

sc_vector* sc_vector_cross_inplace(sc_vector* a, sc_vector* b) {
 if (a->size != 3 || b->size != 3) {
        CCB_ERROR("Vector size mismatch: %u vs %u", a->size, b->size);
        return NULL;
    }

    if (a->type != b->type) {
        CCB_ERROR("Vector type mismatch: %d vs %d", a->type, b->type);
        return NULL;
    }



    switch (a->type) {
        case sc_float16: {
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

        case sc_float32: {
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

        case sc_float64: {
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
            CCB_ERROR("Unsupported sc_TYPES value %d", a->type);
            return NULL;
        }
    }
    return a;
}


sc_value_t norm_p_maper(sc_value_t a, void* b) {

    sc_value_t b_val = *(sc_value_t *)b;

    return sc_scalar_pow(a, b_val);
}


sc_value_t sc_vector_norm(sc_vector* a, uint64_t p, ccb_arena* tmp_arena) {
    if (a->size == 0) {
        CCB_ERROR("Vector size is 0");
        return (sc_value_t){0};
    }

    if (p == 0) {
        CCB_ERROR("p cannot be 0");
        return (sc_value_t){0};
    }


    sc_value_t out = to_sc_value(0.0, a->type);
    sc_vector_reduce(a, sc_scalar_add, out);


    if (p==1) {
        sc_vector* tmp =  sc_vector_map(a, sc_scalar_abs, tmp_arena);
        CCB_NOTNULL(tmp, "Failed to create temporary vector");
        return sc_vector_reduce(tmp, sc_scalar_add, out);

    } else {
        sc_value_t p_val = to_sc_value((float)p, a->type);
        sc_vector* tmp =  sc_vector_map_args(a, norm_p_maper, tmp_arena, &p_val);
        CCB_NOTNULL(tmp, "Failed to create temporary vector");
        return sc_scalar_root(sc_vector_reduce(tmp, sc_scalar_add, out), p_val);
    }
}


sc_vector* sc_vector_normalize(sc_vector* a, ccb_arena* arena) {
    sc_value_t norm = sc_vector_norm(a, 2, arena);
    if (sc_value_to_f64(norm) == 0) {
        CCB_ERROR("Vector norm is 0");
        return NULL;
    }

    return sc_vector_map_args(a, sc_scalar_div_args, arena, &norm);
}

sc_vector* sc_vector_normalize_inplace(sc_vector* a, ccb_arena* arena) {
    sc_value_t norm = sc_vector_norm(a, 2, arena);
    if (sc_value_to_f64(norm) == 0) {
        CCB_ERROR("Vector norm is 0");
        return NULL;
    }
    return sc_vector_map_args_inplace(a, sc_scalar_div_args, &norm);
}





