#include "sc_engine.h"
#include "sc_threads.h" 
#include "const.h"
#include "../ccbase/logs/log.h"
#include "../ccbase/utils/mem.h"

#include <stdlib.h>
#include <stdint.h>


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

int execute_dot_op(void* a, void* b, sc_value_t (*func)(sc_value_t, sc_value_t), sc_TYPES type, uint64_t count, sc_value_t* out) {
    
    CCB_NOT_IMPLEMENTED();
    return -1;
}


// multi thread warpers

void* multi_execute_element_wise_op(void* args) {
    struct thread_data* data = (struct thread_data*)args;
    void* a = data->a;
    void* b = data->b;
    void* out = data->out;

    CCB_NOTNULL(data, "data is NULL");
    CCB_NOTNULL(a, "a is NULL");
    CCB_NOTNULL(b, "b is NULL");
    CCB_NOTNULL(out, "out is NULL");

    int start_delta = data->count / data->thread_count;
    int end_delta = data->count % data->thread_count;
    
    int data_size = 0;
    if (data->type == sc_float16) {
        data_size = sizeof(__bf16);
    } else if (data->type == sc_float32) {
        data_size = sizeof(float);
    } else if (data->type == sc_float64) {
        data_size = sizeof(double);
    } else {
        CCB_ERROR("Unsupported sc_TYPES value %d", data->type);
        return (void*)-1;
    }
    
    void* a_start = (void*)((uintptr_t)a + data->id * start_delta * data_size);
    void* b_start = (void*)((uintptr_t)b + data->id * start_delta * data_size);
    void* out_start = (void*)((uintptr_t)out + data->id * start_delta * data_size);

    if (data->id != data->thread_count - 1) {
        end_delta = start_delta;
    } 

    uint64_t _out = (uint64_t)(a_start, b_start, out_start, data->func, data->type, end_delta);

    lock_mutex(data->mutex);
    data->succes++;
    unlock_mutex(data->mutex);


    return (void*)_out;
}

int multi_execute_scalar_element_op(void* args) {

    struct thread_data* data = (struct thread_data*)args;
    void* a = data->a;
    void* out = data->out;

    CCB_NOTNULL(data, "data is NULL");
    CCB_NOTNULL(a, "a is NULL");
    CCB_NOTNULL(out, "out is NULL");

    int start_delta = data->count / data->thread_count;
    int end_delta = data->count % data->thread_count;
    
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

    return execute_scalar_element_op(a_start, data->scalar, out_start, data->func.scalar_func, data->type, end_delta);
}


int multi_execute_reduce_op(void* args) {
    
    struct thread_data* data = (struct thread_data*)args;
    void* a = data->a;

    CCB_NOTNULL(data, "data is NULL");
    CCB_NOTNULL(a, "a is NULL");


    int start_delta = data->count / data->thread_count;
    int end_delta = data->count % data->thread_count;
    
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
    

    return execute_reduce_op(a_start, data->scalar, data->func.scalar_func, data->type, end_delta, &data->scalar);
}


int multi_execute_map_op(void* args) {
    struct thread_data* data = (struct thread_data*)args;
    void* a = data->a;
    void* out = data->out;

    CCB_NOTNULL(data, "data is NULL");
    CCB_NOTNULL(a, "a is NULL");
    CCB_NOTNULL(out, "out is NULL");

    int start_delta = data->count / data->thread_count;
    int end_delta = data->count % data->thread_count;
    
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
    
    return execute_map_op(a_start, out_start, data->func.scalar_func_map, data->type, end_delta);
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
    int end_delta = data->count % data->thread_count;
    
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

    return execute_map_args_op(start_a, start_out, data->func.scalar_func_map_args, data->type, end_delta, _args);
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
                ((sc_vector*)task->out)->type = ((sc_vector*)task->a)->type;
                ((sc_vector*)task->out)->size = ((sc_vector*)task->a)->size;
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
            out_code = execute_element_wise_op(a, b, task->out, task->task_func.scalar_func, type, task->opration_count);
            break;

        case sc_element_scalar_op:
            CCB_NOTNULL(task->out, "task->out is NULL for element scalar operation");
            out_code = execute_scalar_element_op(a, task->scalar, task->out, task->task_func.scalar_func, type, task->opration_count);
            break;

        case sc_reduce_op:
            CCB_NOTNULL(task->out, "task->out is NULL for reduce operation");
            out_code = execute_reduce_op(a, task->scalar, task->task_func.scalar_func, type, task->opration_count, &out->scalar_result);
            break;
        
        case sc_map_op:
            CCB_NOTNULL(task->out, "task->out is NULL for map operation");
            out_code = execute_map_op(a, task->out, task->task_func.scalar_func_map, type, task->opration_count);
            break;

        case sc_map_args_op:
            CCB_NOTNULL(task->out, "task->out is NULL for map args operation");
            out_code = execute_map_args_op(a, task->out, task->task_func.scalar_func_map_args, type, task->opration_count, task->args);
            break;

        case sc_dot_op:
            CCB_NOTNULL(task->b, "task->b is NULL for dot operation");
            CCB_NOTNULL(task->out, "task->out is NULL for dot operation");
            out_code = execute_dot_op(a, b, task->task_func.scalar_func, type, task->opration_count, &out->scalar_result);
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
            ((sc_vector*)task->out)->data = out_data;
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
    int thread_count = get_cpu_count();
    
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
    struct thread_data data = {
        .a = a,
        .b = b,
        .out = out_data,
        .args = task->args,
        .scalar = task->scalar,
        .data_type = data_type,
        .func = task->task_func,
        .type = type,
        .count = task->opration_count,
        .thread_count = get_cpu_count(),
        .mutex = NULL,
        .succes = 0,
        .id = 0,
    };

    create_mutex(&data.mutex);


    // lanch threads
    thread_t* threads = (thread_t*)malloc(thread_count * sizeof(thread_t));


    for (uint64_t i = 0; i < thread_count; i++) {
        data.id = i;
        
        switch (task->op_type) {
            case sc_element_wise_op:
                create_thread(&threads[i], multi_execute_element_wise_op, &data);
                break;

            

            default:
                CCB_ERROR("Unsupported operation type %d", task->op_type);
                return out;
        }
    }

    for (uint64_t i = 0; i < thread_count; i++) {
        if (join_thread(threads[i]) == -1) {
            CCB_ERROR("Failed to join thread %d", i);
            out->succes = 0;
            return out;
        }
    }

    if (data.succes != thread_count) {
        CCB_ERROR("Failed to execute multi thread operation: %d/%d", data.succes, thread_count);
        out->succes = 0;
        return out;
    }

    out->succes = 1;
    
    switch (task->data_type) {
        case sc_vector_type:
            ((sc_vector*)task->out)->data = out_data;
            break;

        case sc_tensor_type:
            ((sc_vector*)task->out)->data = out_data;
            break;

        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", task->data_type);
            return out;
    }

    destroy_mutex(data.mutex);
    free(threads);

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
 