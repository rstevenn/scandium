#include "sc_engine.h"
#include "sc_threads.h" 
#include "const.h"
#include "ccbase/logs/log.h"
#include "ccbase/utils/mem.h"

#include <stdlib.h>
#include <stdint.h>
#include <immintrin.h>

struct thread_data {
    int succes;
    void* a;
    void* b;
    void* out;
    void* args;
    sc_value_t scalar;
    sc_engine_func func;
    sc_TYPES type;
    sc_engine_data_type data_type;
    uint64_t count;
    uint64_t id;
    uint64_t thread_count;
    mutex_t mutex;
};

struct thread_control {
    mutex_t run_mutex;
    mutex_t wait_mutex;
    struct thread_data* task_data;
    int (*task_fn)(void*);
    int return_value;
    ccb_arena* arena;
};


// thread pool
thread_t* threads = NULL;
struct thread_control* thread_controls = NULL;
uint64_t thread_count = 0;


void* sc_worker(void* arg);
void sc_init_thread_pool(uint64_t num_threads) {
    
    if (threads != NULL) {
        return;
    }
    
    if (num_threads == 0) {
        num_threads = get_cpu_count();
    }

    threads = malloc(num_threads*sizeof(thread_t));
    if (threads == NULL) {
        CCB_ERROR("Failed to allocate memory for thread pool");
        return;
    }
    
    thread_controls = malloc(num_threads*sizeof(struct thread_control));
    if (thread_controls == NULL) {
        CCB_ERROR("Failed to allocate memory for thread control");
        free(threads);
        threads = NULL;
        return;
    }

    for (uint64_t i = 0; i < num_threads; i++) {
        create_mutex(&thread_controls[i].run_mutex);
        create_mutex(&thread_controls[i].wait_mutex);
        lock_mutex(thread_controls[i].wait_mutex);
        thread_controls[i].task_data = NULL;
    
        create_thread(&threads[i], sc_worker, &thread_controls[i]);
        if (threads[i] == 0) {
            CCB_ERROR("Failed to create thread %d", i);
            exit(1);
        }
        unlock_mutex(thread_controls[i].run_mutex);   
    }


    thread_count = num_threads;
}

void sc_destroy_thread_pool() {
    if (threads != NULL) {
        free(threads);
        threads = NULL;
        thread_count = 0;
    }
}

void* sc_worker(void* arg) {
    struct thread_control* data = (struct thread_control*)arg;

    while(1) {
        lock_mutex(data->wait_mutex);
        lock_mutex(data->run_mutex);
        unlock_mutex(data->wait_mutex);

        if (data->task_data == NULL) {
            unlock_mutex(data->run_mutex);
            break;
        }
        data->return_value = data->task_fn(data->task_data);
        data->task_data = NULL;
        unlock_mutex(data->run_mutex);
                
    }
}


// AVX optimized functions
int element_wise_avx_f32(float* a, float* b, float* out, sc_value_t (*func)(sc_value_t, sc_value_t), uint64_t count) {

    if (func == sc_scalar_add) {
        for (uint64_t i = 0; i < count; i+=8){
            __m256 x, y, _out;

            if (i + 8 > count) break;

            x = _mm256_loadu_ps(&a[i]);
            y = _mm256_loadu_ps(&b[i]);
            _out = _mm256_add_ps(x, y);
            _mm256_storeu_ps(&out[i], _out);
         
        }

        uint64_t base = count - count%8;
        for (uint64_t i = 0; i < count%8; i++) {
            out[base+i] = func(to_sc_value(a[base+i], sc_float32), to_sc_value(b[base+i], sc_float32)).value.f32;
        }

        return 0;
    
    } else if (func == sc_scalar_sub) {
    
        for (uint64_t i = 0; i < count; i+=8){
            __m256 x, y, _out;

            if (i + 8 > count) break;

            x = _mm256_loadu_ps(&a[i]);
            y = _mm256_loadu_ps(&b[i]);
            _out = _mm256_sub_ps(x, y);
            _mm256_storeu_ps(&out[i], _out);
         
        }

        uint64_t base = count - count%8;
        for (uint64_t i = 0; i < count%8; i++) {
            out[base+i] = func(to_sc_value(a[base+i], sc_float32), to_sc_value(b[base+i], sc_float32)).value.f32;
        }

        return 0;

    } else if (func == sc_scalar_mul) {
    
        for (uint64_t i = 0; i < count; i+=8){
            __m256 x, y, _out;

            if (i + 8 > count) break;

            x = _mm256_loadu_ps(&a[i]);
            y = _mm256_loadu_ps(&b[i]);
            _out = _mm256_mul_ps(x, y);
            _mm256_storeu_ps(&out[i], _out);
         
        }

        uint64_t base = count - count%8;
        for (uint64_t i = 0; i < count%8; i++) {
            out[base+i] = func(to_sc_value(a[base+i], sc_float32), to_sc_value(b[base+i], sc_float32)).value.f32;
        }

        return 0;
    
    } else if (func == sc_scalar_div) {
    
        for (uint64_t i = 0; i < count; i+=8){
            __m256 x, y, _out;

            if (i + 8 > count) break;

            x = _mm256_loadu_ps(&a[i]);
            y = _mm256_loadu_ps(&b[i]);
            _out = _mm256_div_ps(x, y);
            _mm256_storeu_ps(&out[i], _out);
         
        }

        uint64_t base = count - count%8;
        for (uint64_t i = 0; i < count%8; i++) {
            out[base+i] = func(to_sc_value(a[base+i], sc_float32), to_sc_value(b[base+i], sc_float32)).value.f32;
        }

        return 0;

    } 


    return -1;
}


int map_avx_f32(float* a, float* out, sc_value_t (*func)(sc_value_t), uint64_t count) {

    if (func == sc_scalar_abs) {
        for (uint64_t i = 0; i < count; i+=8){
            __m256 x, _out;

            if (i + 8 > count) break;

            x = _mm256_loadu_ps(&a[i]);
            _out = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x); // Clear sign bit
            _mm256_storeu_ps(&out[i], _out);
         
        }

        uint64_t base = count - count%8;
        for (uint64_t i = 0; i < count%8; i++) {
            out[base+i] = func(to_sc_value(a[base+i], sc_float32)).value.f32;
        }

        return 0;
    
    } 
    return -1;
}

int map_args_avx_f32(float* a, float* out, sc_value_t (*func)(sc_value_t, void*), void* args, uint64_t count) {

    if (func == sc_scalar_add_args) {
        sc_value_t b = *(sc_value_t *)args;
        for (uint64_t i = 0; i < count; i+=8){
            __m256 x, y, _out;
            float b_f = b.value.f32;

            if (i + 8 > count) break;

            x = _mm256_loadu_ps(&a[i]);
            y = _mm256_set1_ps(b_f);
            _out = _mm256_add_ps(x, y);
            _mm256_storeu_ps(&out[i], _out);
         
        }

        uint64_t base = count - count%8;
        for (uint64_t i = 0; i < count%8; i++) {
            out[base+i] = func(to_sc_value(a[base+i], sc_float32), args).value.f32;
        }

        return 0;
    
    } else if (func == sc_scalar_sub_args) {
        sc_value_t b = *(sc_value_t *)args;
        for (uint64_t i = 0; i < count; i+=8){
            __m256 x, y, _out;
            float b_f = b.value.f32;

            if (i + 8 > count) break;

            x = _mm256_loadu_ps(&a[i]);
            y = _mm256_set1_ps(b_f);
            _out = _mm256_sub_ps(x, y);
            _mm256_storeu_ps(&out[i], _out);
         
        }

        uint64_t base = count - count%8;
        for (uint64_t i = 0; i < count%8; i++) {
            out[base+i] = func(to_sc_value(a[base+i], sc_float32), args).value.f32;
        }

        return 0;

    } else if (func == sc_scalar_mul_args) {
        sc_value_t b = *(sc_value_t *)args;
        for (uint64_t i = 0; i < count; i+=8){
            __m256 x, y, _out;
            float b_f = b.value.f32;

            if (i + 8 > count) break;

            x = _mm256_loadu_ps(&a[i]);
            y = _mm256_set1_ps(b_f);
            _out = _mm256_mul_ps(x, y);
            _mm256_storeu_ps(&out[i], _out);
         
        }

        uint64_t base = count - count%8;
        
        for (uint64_t i = 0; i < count%8; i++) {
            out[base+i] = func(to_sc_value(a[base+i], sc_float32), args).value.f32;
        }

        return 0;
    
    } else if (func == sc_scalar_div_args) {

        sc_value_t b = *(sc_value_t *)args;
        for (uint64_t i = 0; i < count; i+=8){
            __m256 x, y, _out;
            float b_f = b.value.f32;

            if (i + 8 > count) break;

            x = _mm256_loadu_ps(&a[i]);
            y = _mm256_set1_ps(b_f);
            _out = _mm256_div_ps(x, y);
            _mm256_storeu_ps(&out[i], _out);
         
        }

        uint64_t base = count - count%8;
        for (uint64_t i = 0; i < count%8; i++) {
            out[base+i] = func(to_sc_value(a[base+i], sc_float32), args).value.f32;
        }

        return 0;

    } 

    return -1;
}


// single thread functions
int execute_element_wise_op(void* a, void* b, void* out, sc_value_t (*func)(sc_value_t, sc_value_t), sc_TYPES type, uint64_t count) {
    
    switch (type) {
        case sc_float16: {
            __bf16* a_data = (__bf16*)a;
            __bf16* b_data = (__bf16*)b;
            __bf16* out_data = (__bf16*)out;

            for (uint64_t i = 0; i < count; i++) {
                out_data[i] = func(to_sc_value((float)a_data[i], sc_float16),
                                   to_sc_value((float)b_data[i], sc_float16)).value.f16;
            }
            break;
        }

        case sc_float32: {
            float* a_data = (float*)a;
            float* b_data = (float*)b;
            float* out_data = (float*)out;


            if (__builtin_cpu_supports("avx") >= 256) {
                if (element_wise_avx_f32(a_data, b_data, out_data, func, count) == 0) {
                    return 0;
                }
            }

            for (uint64_t i = 0; i < count; i++) {
                out_data[i] = func(to_sc_value(a_data[i], sc_float32),
                                   to_sc_value(b_data[i], sc_float32)).value.f32;
            }
            break;
        }

        case sc_float64: {
            double* a_data = (double*)a;
            double* b_data = (double*)b;
            double* out_data = (double*)out;

            for (uint64_t i = 0; i < count; i++) {
                out_data[i] = func(to_sc_value(a_data[i], sc_float64),
                                   to_sc_value(b_data[i], sc_float64)).value.f64;
            }
            break;
        }

        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", type);
            return -1;
    }

    return 0;
}


int execute_scalar_element_op(void* a, sc_value_t scalar, void* out, sc_value_t (*func)(sc_value_t, sc_value_t), sc_TYPES type, uint64_t count) {
    switch (type) {
        case sc_float16: {
            __bf16* a_data = (__bf16*)a;
            __bf16* out_data = (__bf16*)out;

            for (uint64_t i = 0; i < count; i++) {
                out_data[i] = func(to_sc_value((float)a_data[i], sc_float16), scalar).value.f16;
            }
            break;
        }

        case sc_float32: {
            float* a_data = (float*)a;
            float* out_data = (float*)out;

            for (uint64_t i = 0; i < count; i++) {
                out_data[i] = func(to_sc_value(a_data[i], sc_float32), scalar).value.f32;
            }
            break;
        }

        case sc_float64: {
            double* a_data = (double*)a;
            double* out_data = (double*)out;

            for (uint64_t i = 0; i < count; i++) {
                out_data[i] = func(to_sc_value(a_data[i], sc_float64), scalar).value.f64;
            }
            break;
        }

        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", type);
            return -1;
    }

    return 0;
}


int execute_reduce_op(void* a, sc_value_t init_val, sc_value_t (*func)(sc_value_t, sc_value_t), sc_TYPES type, uint64_t count, sc_value_t* out) {
    *out = init_val;

    switch (type) {
        case sc_float16: {
            __bf16* a_data = (__bf16*)a;

            for (uint64_t i = 0; i < count; i++) {
                *out = func(*out, to_sc_value((float)a_data[i], sc_float16));
            }
            break;
        }

        case sc_float32: {
            float* a_data = (float*)a;

            for (uint64_t i = 0; i < count; i++) {
                *out = func(*out, to_sc_value(a_data[i], sc_float32));
            }
            break;
        }

        case sc_float64: {
            double* a_data = (double*)a;

            for (uint64_t i = 0; i < count; i++) {
                *out = func(*out, to_sc_value(a_data[i], sc_float64));
            }
            break;
        }

        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", type);
            return -1;
    }

    return 0;
}

int execute_map_op(void* a, void* out, sc_value_t (*func)(sc_value_t), sc_TYPES type, uint64_t count) {
    switch (type) {
        case sc_float16: {
            __bf16* a_data = (__bf16*)a;
            __bf16* out_data = (__bf16*)out;

            for (uint64_t i = 0; i < count; i++) {
                out_data[i] = func(to_sc_value((float)a_data[i], sc_float16)).value.f16;
            }
            break;
        }

        case sc_float32: {
            float* a_data = (float*)a;
            float* out_data = (float*)out;

            if (__builtin_cpu_supports("avx") >= 256) {
                if (map_avx_f32(a_data, out_data, func, count) == 0) {
                    return 0;
                }
            }

            for (uint64_t i = 0; i < count; i++) {
                out_data[i] = func(to_sc_value(a_data[i], sc_float32)).value.f32;
            }
            break;
        }

        case sc_float64: {
            double* a_data = (double*)a;
            double* out_data = (double*)out;

            for (uint64_t i = 0; i < count; i++) {
                out_data[i] = func(to_sc_value(a_data[i], sc_float64)).value.f64;
            }
            break;
        }

        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", type);
            return -1;
    }

    return 0;
}

int execute_map_args_op(void* a, void* out, sc_value_t (*func)(sc_value_t, void*), sc_TYPES type, uint64_t count, void* args) {
    switch (type) {
        case sc_float16: {
            __bf16* a_data = (__bf16*)a;
            __bf16* out_data = (__bf16*)out;

            for (uint64_t i = 0; i < count; i++) {
                out_data[i] = func(to_sc_value((float)a_data[i], sc_float16), args).value.f16;
            }
            break;
        }

        case sc_float32: {
            float* a_data = (float*)a;
            float* out_data = (float*)out;

            if (__builtin_cpu_supports("avx") >= 256) {
                if (map_args_avx_f32(a_data, out_data, func, args, count) == 0) {
                    return 0;
                }
            }

            for (uint64_t i = 0; i < count; i++) {
                out_data[i] = func(to_sc_value(a_data[i], sc_float32), args).value.f32;
            }
            break;
        }

        case sc_float64: {
            double* a_data = (double*)a;
            double* out_data = (double*)out;

            for (uint64_t i = 0; i < count; i++) {
                out_data[i] = func(to_sc_value(a_data[i], sc_float64), args).value.f64;
            }
            break;
        }

        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", type);
            return -1;
    }

    return 0;
}


// multi thread warpers

int multi_execute_element_wise_op(void* args) {
    struct thread_data* data = (struct thread_data*)args;
    void* a = data->a;
    void* b = data->b;
    void* out = data->out;

    CCB_NOTNULL(data, "data is NULL");
    CCB_NOTNULL(a, "a is NULL");
    CCB_NOTNULL(b, "b is NULL");
    CCB_NOTNULL(out, "out is NULL");

    int start_delta = (data->count) / (float)data->thread_count;
    int end_delta = start_delta + data->count % data->thread_count;
    
    int data_size = 0;
    if (data->type == sc_float16) {
        data_size = sizeof(__bf16);
    } else if (data->type == sc_float32) {
        data_size = sizeof(float);
    } else if (data->type == sc_float64) {
        data_size = sizeof(double);
    } else {
        CCB_ERROR("Unsupported sc_TYPES value %d", data->type);
        return -1;
    }

    if (data->id != data->thread_count - 1) {
        end_delta = start_delta;
    } 


    void* a_start = (void*)((uintptr_t)a + data->id * start_delta * data_size);
    void* b_start = (void*)((uintptr_t)b + data->id * start_delta * data_size);
    void* out_start = (void*)((uintptr_t)out + data->id * start_delta * data_size);
    
    
    uint64_t _out = (uint64_t)execute_element_wise_op(a_start, b_start, out_start, data->func.scalar_func, data->type, end_delta);

    if (_out == 0) {
        lock_mutex(data->mutex);
        data->succes++;
        unlock_mutex(data->mutex);
    }

    return _out;
}

int multi_execute_scalar_element_op(void* args) {

    struct thread_data* data = (struct thread_data*)args;
    void* a = data->a;
    void* out = data->out;

    CCB_NOTNULL(data, "data is NULL");
    CCB_NOTNULL(a, "a is NULL");
    CCB_NOTNULL(out, "out is NULL");

    int start_delta = data->count / data->thread_count;
    int end_delta = start_delta + data->count % data->thread_count;
    
    int data_size = 0;
    if (data->type == sc_float16) {
        data_size = sizeof(__bf16);
    } else if (data->type == sc_float32) {
        data_size = sizeof(float);
    } else if (data->type == sc_float64) {
        data_size = sizeof(double);
    } else {
        CCB_ERROR("Unsupported sc_TYPES value %d", data->type);
        return -1;
    }
    
    void* a_start = (void*)((uintptr_t)a + data->id * start_delta * data_size);
    void* out_start = (void*)((uintptr_t)out + data->id * start_delta * data_size);

    if (data->id != data->thread_count - 1) {
        end_delta = start_delta;
    } 

    uint64_t return_code = execute_scalar_element_op(a_start, data->scalar, out_start, data->func.scalar_func, data->type, end_delta); 
    
    if (return_code == 0) {
        lock_mutex(data->mutex);
        data->succes++;
        unlock_mutex(data->mutex);
    }

    return return_code;
}


int multi_execute_reduce_op(void* args) {
    
    struct thread_data* data = (struct thread_data*)args;
    void* a = data->a;

    CCB_NOTNULL(data, "data is NULL");
    CCB_NOTNULL(a, "a is NULL");


    int start_delta = data->count / data->thread_count;
    int end_delta = start_delta + data->count % data->thread_count;
    
    int data_size = 0;
    if (data->type == sc_float16) {
        data_size = sizeof(__bf16);
    } else if (data->type == sc_float32) {
        data_size = sizeof(float);
    } else if (data->type == sc_float64) {
        data_size = sizeof(double);
    } else {
        CCB_ERROR("Unsupported sc_TYPES value %d", data->type);
        return -1;
    }

    void* a_start = (void*)((uintptr_t)a + data->id * start_delta * data_size);
    
    if (data->id != data->thread_count - 1) {
        end_delta = start_delta;
    } 
    
    uint64_t return_code = execute_reduce_op(a_start, data->scalar, data->func.scalar_func, data->type, end_delta, &data->scalar); 

    if (return_code == 0) {
        lock_mutex(data->mutex);
        data->succes++;
        unlock_mutex(data->mutex);
    }

    return return_code;
}



int multi_execute_map_op(void* args) {
    struct thread_data* data = (struct thread_data*)args;
    void* a = data->a;
    void* out = data->out;

    CCB_NOTNULL(data, "data is NULL");
    CCB_NOTNULL(a, "a is NULL");
    CCB_NOTNULL(out, "out is NULL");

    int start_delta = data->count / data->thread_count;
    int end_delta = start_delta+ data->count % data->thread_count;
    
    int data_size = 0;
    if (data->type == sc_float16) {
        data_size = sizeof(__bf16);
    } else if (data->type == sc_float32) {
        data_size = sizeof(float);
    } else if (data->type == sc_float64) {
        data_size = sizeof(double);
    } else {
        CCB_ERROR("Unsupported sc_TYPES value %d", data->type);
        return -1;
    }

    void* a_start = (void*)((uintptr_t)a + data->id * start_delta * data_size);
    void* out_start = (void*)((uintptr_t)out + data->id * start_delta * data_size);

    if (data->id != data->thread_count - 1) {
        end_delta = start_delta;
    } 
    
    uint64_t return_code = execute_map_op(a_start, out_start, data->func.scalar_func_map, data->type, end_delta);

    if (return_code == 0) {
        lock_mutex(data->mutex);
        data->succes++;
        unlock_mutex(data->mutex);
    }

    return return_code;
}


int multi_execute_map_args_op(void* args) {
    struct thread_data* data = (struct thread_data*)args;
    void* a = data->a;
    void* out = data->out;
    void* _args = data->args;

    CCB_NOTNULL(data, "data is NULL");
    CCB_NOTNULL(a, "a is NULL");
    CCB_NOTNULL(out, "out is NULL");
    CCB_NOTNULL(_args, "args is NULL");

    int start_delta = data->count / data->thread_count;
    int end_delta = start_delta+data->count % data->thread_count;
    
    int data_size = 0;
    if (data->type == sc_float16) {
        data_size = sizeof(__bf16);
    } else if (data->type == sc_float32) {
        data_size = sizeof(float);
    } else if (data->type == sc_float64) {
        data_size = sizeof(double);
    } else {
        CCB_ERROR("Unsupported sc_TYPES value %d", data->type);
        return -1;
    }

    void* start_a = (void*)((uintptr_t)a + data->id * start_delta * data_size);
    void* start_out = (void*)((uintptr_t)out + data->id * start_delta * data_size);

    if (data->id != data->thread_count - 1) {
        end_delta = start_delta;
    } 

    uint64_t return_code = execute_map_args_op(start_a, start_out, data->func.scalar_func_map_args, data->type, end_delta, _args);

    if (return_code == 0) {
        lock_mutex(data->mutex);
        data->succes++;
        unlock_mutex(data->mutex);
    }

    return return_code;
}


// engine functions
sc_task_result* execute_single_thread(sc_task* task, sc_task_result* out) {
    
    void* a  = NULL;
    void* b  = NULL;
    void* out_data = NULL;
    sc_TYPES type;

    out->succes = 0;
    CCB_NOTNULL(task->a, "task->a is NULL");
    CCB_NOTNULL(task->task_func.scalar_func, "task->task_func is NULL");
    
    
    switch (task->data_type) {

        case sc_vector_type:
            a = ((sc_vector*)task->a)->data;
            
            if (task->b != NULL) {
                b = ((sc_vector*)task->b)->data;
            } 

            if (task->out != NULL) {
                out_data = ((sc_vector*)task->out)->data;
            }
            
            type = ((sc_vector*)task->a)->type;
            

            break;


        case sc_tensor_type:
            CCB_ERROR("Not implemented, datatype: %d", task->data_type);
            return out;


        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", task->data_type);
            return out;
    }

    int out_code = 0;
    switch (task->op_type) {
        case sc_element_wise_op:
            CCB_NOTNULL(task->b, "task->b is NULL for element wise operation");
            CCB_NOTNULL(task->out, "task->out is NULL for element wise operation");
            out_code = execute_element_wise_op(a, b, out_data, task->task_func.scalar_func, type, task->opration_count);
            break;

        case sc_element_scalar_op:
            CCB_NOTNULL(task->out, "task->out is NULL for element scalar operation");
            out_code = execute_scalar_element_op(a, task->scalar, out_data, task->task_func.scalar_func, type, task->opration_count);
            break;

        case sc_reduce_op:
            out_code = execute_reduce_op(a, task->scalar, task->task_func.scalar_func, type, task->opration_count, &out->scalar_result);
            break;
        
        case sc_map_op:
            CCB_NOTNULL(task->out, "task->out is NULL for map operation");
            out_code = execute_map_op(a, out_data, task->task_func.scalar_func_map, type, task->opration_count);
            break;

        case sc_map_args_op:
            CCB_NOTNULL(task->out, "task->out is NULL for map args operation");
            out_code = execute_map_args_op(a, out_data, task->task_func.scalar_func_map_args, type, task->opration_count, task->args);
            break;

        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", task->op_type);
            return out;
    }
    
    if (out_code != 0) {
        CCB_ERROR("Failed to execute operation");
        out->succes = 0;
        return out;
    }

    out->succes = 1;

    switch (task->data_type) {
        case sc_vector_type:
            if (task->op_type != sc_reduce_op){
                ((sc_vector*)task->out)->data = out_data;
            } 
            break;

        case sc_tensor_type:
            ((sc_vector*)task->out)->data = out_data;
            break;

        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", task->data_type);
            return out;
    }
    return out;
}



sc_task_result* execute_multi_thread(sc_task* task, sc_task_result* out) {
    sc_init_thread_pool(0);


    int thread_count = get_cpu_count();
    thread_count = min(thread_count, task->opration_count);
    
    void* a  = NULL;
    void* b  = NULL;
    void* out_data = NULL;
    sc_TYPES type;
    sc_engine_data_type data_type;


    out->succes = 0;

    // retreive data
    switch (task->data_type) {

        case sc_vector_type:
            a = ((sc_vector*)task->a)->data;
            
            if (task->b != NULL) {
                b = ((sc_vector*)task->b)->data;
            } 

            if (task->out != NULL) {
                out_data = ((sc_vector*)task->out)->data;
            }

            type = ((sc_vector*)task->a)->type;
            data_type = sc_vector_type;
            
            break;


        case sc_tensor_type:
            CCB_NOT_IMPLEMENTED();
            break;

        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", task->data_type);
            return out;
    }

    // configure threads
    struct thread_data* data = (struct thread_data*)malloc(thread_count * sizeof(struct thread_data));
    CCB_NOTNULL(data, "Failed to allocate thread data");
    


    mutex_t mutex;
    create_mutex(&mutex);


    // lanch threads
    for (uint64_t i = 0; i < thread_count; i++) {
        data[i].a = a;
        data[i].b = b;
        data[i].out = out_data;
        data[i].args = task->args;
        data[i].scalar = task->scalar;
        data[i].data_type = data_type;
        data[i].func = task->task_func;
        data[i].type = type;
        data[i].count = task->opration_count;
        data[i].thread_count = thread_count;
        data[i].mutex = mutex;
        data[i].succes = 0;
        data[i].id = i;
        
        switch (task->op_type) {
            case sc_element_wise_op:
                thread_controls[i].task_data = &data[i];
                thread_controls[i].task_fn = multi_execute_element_wise_op;
                unlock_mutex(thread_controls[i].wait_mutex);
                break;

            case sc_element_scalar_op:
                thread_controls[i].task_data = &data[i];
                thread_controls[i].task_fn = multi_execute_scalar_element_op;
                unlock_mutex(thread_controls[i].wait_mutex);
                break;

            case sc_reduce_op:
                thread_controls[i].task_data = &data[i];
                thread_controls[i].task_fn = multi_execute_reduce_op;
                unlock_mutex(thread_controls[i].wait_mutex);
                break;

            case sc_map_op:
                thread_controls[i].task_data = &data[i];
                thread_controls[i].task_fn = multi_execute_map_op;
                unlock_mutex(thread_controls[i].wait_mutex);
                break;

            case sc_map_args_op:
                thread_controls[i].task_data = &data[i];
                thread_controls[i].task_fn = multi_execute_map_args_op;
                unlock_mutex(thread_controls[i].wait_mutex);
                break;

            default:
                CCB_ERROR("Unsupported sc_TYPES value %d", task->op_type);
                break;
        }
    }

    for (uint64_t i = 0; i < thread_count; i++) {
        lock_mutex(thread_controls[i].wait_mutex);
    }

    int rate = 0;
    for (uint64_t i = 0; i < thread_count; i++) {
        lock_mutex(thread_controls[i].run_mutex);
        unlock_mutex(thread_controls[i].run_mutex);

        if (thread_controls[i].return_value != 0) {
            CCB_ERROR("Failed to join thread %d", i);
            out->succes = 0;
            return out;
        }
        rate += data[i].succes;
    }
    
    if (rate != thread_count) {
        CCB_ERROR("Failed to execute multi thread operation: %d/%d", rate, thread_count);
        out->succes = 0;
        return out;
    }

    out->succes = 1;

    if (task->op_type == sc_reduce_op) {
        out->scalar_result = task->scalar;
        for (uint64_t i = 0; i < thread_count; i++) {
            out->scalar_result = task->task_func.scalar_func(out->scalar_result, data[i].scalar);
        }
    }

    switch (task->data_type) {
        case sc_vector_type:
            if (task->op_type != sc_reduce_op){
                ((sc_vector*)task->out)->data = out_data;
            }
            break;

        case sc_tensor_type:
            ((sc_vector*)task->out)->data = out_data;
            break;

        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", task->data_type);
            return out;
    }

    destroy_mutex(mutex);
    free(data);
    
    return out;
   
}



sc_task* sc_create_task(sc_engine_data_type data_type, 
                        sc_engine_op_type op_type, 
                        void* a, void* b, void* out, sc_value_t scalar, void* args, 
                        sc_engine_func task_func, uint64_t opration_count, ccb_arena* arena){
    
    CCB_NOTNULL(arena, "arena is NULL");
    sc_task* task = (sc_task*)ccb_arena_malloc(arena, sizeof(sc_task));
    CCB_NOTNULL(task, "Failed to allocate task");

    task->data_type = data_type;
    task->op_type = op_type;
    task->a = a;
    task->b = b;
    task->out = out;
    task->scalar = scalar;
    task->args = args;
    task->task_func = task_func;
    task->opration_count = opration_count;

    return task;
}



sc_task_result* sc_execute_task(sc_task* task, sc_execution_mode mode, sc_task_result* out, ccb_arena* arena) {
    CCB_NOTNULL(task, "task is NULL");
    CCB_NOTNULL(out, "out is NULL");

    out->succes = 0;

    sc_execution_mode exec_mode = mode;
    if (exec_mode == sc_auto) {
        if (task->opration_count > MULTITHRAD_OPRATION_TRESHOLD) {
            exec_mode = sc_multi_thread;
        } else {
            exec_mode = sc_single_thread;
        }
    }
    
    switch (exec_mode) {
        case sc_single_thread:
            return execute_single_thread(task, out);
        case sc_multi_thread:
            return execute_multi_thread(task, out);
        default:
            CCB_ERROR("Unsupported execution mode %d", exec_mode);
            return NULL;
    }
}
 
