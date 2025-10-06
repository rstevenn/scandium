#ifndef __SC_ENGINE_H__
#define __SC_ENGINE_H__

#include "data.h"
#include "linalg.h"
#include "ccbase/utils/mem.h"

// if the operation size is above this treshold it will be executed in multiple threads
#define MULTITHRAD_OPRATION_TRESHOLD 1024

/*
    the scandium execution engine
    it handels multithreading and the execution pipline
*/

typedef enum {
    sc_vector_type,
    sc_tensor_type
} sc_engine_data_type;

typedef enum {
    sc_element_wise_op,
    sc_element_scalar_op,
    sc_reduce_op,
    sc_map_op,
    sc_map_args_op,
    sc_dot_op
} sc_engine_op_type;

typedef enum {
    sc_auto,
    sc_single_thread,
    sc_multi_thread
} sc_execution_mode;

typedef union {
    sc_value_t (*scalar_func)(sc_value_t, sc_value_t);
    sc_value_t (*scalar_func_map)(sc_value_t);
    sc_value_t (*scalar_func_map_args)(sc_value_t, void*);
} sc_engine_func;



typedef struct {
    sc_engine_data_type data_type;
    sc_engine_op_type op_type;

    void* a;
    void* b;
    void* out;
    sc_value_t scalar;
    void* args;
    
    sc_engine_func task_func;
    
    uint64_t opration_count;
} sc_task;

typedef struct {
    uint64_t succes;
    sc_value_t scalar_result;
    void* result;
} sc_task_result;


sc_task* sc_create_task(sc_engine_data_type data_type, sc_engine_op_type op_type, void* a, void* b, void* out, sc_value_t scalar, void* args, sc_engine_func task_func, uint64_t opration_count, ccb_arena* arena);

#define sc_create_vector_element_wise_task(a, b, out, func, count, arena) sc_create_task(sc_vector_type, sc_element_wise_op, a, b, out, (sc_value_t){0}, NULL, (sc_engine_func){.scalar_func=func}, count, arena)
#define sc_create_vector_scalar_task(a, scalar, out, func, count, arena) sc_create_task(sc_vector_type, sc_element_scalar_op, a, NULL, out, scalar, NULL, (sc_engine_func){.scalar_func=func}, count, arena)
#define sc_create_vector_reduce_task(a, scalar, func, count, arena) sc_create_task(sc_vector_type, sc_reduce_op, a, NULL, NULL, scalar, NULL, (sc_engine_func){.scalar_func=func}, count, arena)
#define sc_create_vector_map_task(a, out, func, count, arena) sc_create_task(sc_vector_type, sc_map_op, a, NULL, out, (sc_value_t){0}, NULL, (sc_engine_func){.scalar_func_map=func}, count, arena)
#define sc_create_vector_map_args_task(a, out, func, args, count, arena) sc_create_task(sc_vector_type, sc_map_args_op, a, NULL, out, (sc_value_t){0}, args, (sc_engine_func){.scalar_func_map_args=func}, count, arena)

sc_task_result* sc_execute_task(sc_task* task, sc_execution_mode mode, sc_task_result* result, ccb_arena* arena);




#endif // __SC_ENGINE_H__