#include "scandium.h"

int main(){
    // init logs and memory
    ccb_InitLog("logs.log");
    ccb_arena* arena = ccb_init_arena();
    
    // create data
    float a[10] = {1,  2, 3, 4, 5, 6, 7, 8, 9, 10};
    float b[10] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

    // create vectors
    sc_vector* vec1 = sc_create_vector(10, sc_float32, arena);
    sc_vector* vec2 = sc_create_vector(10, sc_float32, arena);
    
    // load data into vector
    sc_data_to_vector(vec1, a, 10);
    sc_data_to_vector(vec2, b, 10);

    // execute operations
    sc_vector* result = sc_vector_sub(vec1, vec2, arena);
    result = sc_vector_map_inplace(result, sc_scalar_abs);
    result = sc_vector_div_scalar_inplace(result, to_sc_value(3, sc_float32));

    // print result
    sc_print_vector(result);
    
    // free memory and close logs
    ccb_arena_free(arena);
    ccb_CloseLogFile();

    return 0;
}