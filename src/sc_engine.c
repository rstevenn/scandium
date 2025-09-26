#include "sc_engine.h"
#include "const.h"
#include "../ccbase/logs/log.h"
#include "../ccbase/utils/mem.h"

#include <stdlib.h>
#include <stdint.h>
#include <threads.h>

sc_task* sc_create_task(sc_engine_data_type data_type, 
                        sc_engine_op_type op_type, 
                        void* a, void* b, void* out, sc_value_t scalar, void* args, 
                        sc_engine_func task_func, uint64_t opration_count, ccb_arena* arena){
    
    ccb_notnull(arena, "arena is NULL");
    sc_task* task = (sc_task*)ccb_arena_alloc(arena, sizeof(sc_task));
    ccb_notnull(task, "Failed to allocate task");

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

