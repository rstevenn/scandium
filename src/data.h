#ifndef __DATA_H__
#define __DATA_H__

#include <stdint.h>
#include "../ccbase/utils/mem.h"


// define f16

#define GCC_COMPILER (defined(__GNUC__) && !defined(__clang__))
#if !GCC_COMPILER
    #define __bf16 uint16_t 
#endif


// data structures
typedef enum {
    CL_float16,
    CL_float32,
    CL_float64,
} CL_TYPES;


typedef struct cl_vector_t {
    void* data;
    uint32_t size;
    CL_TYPES type;
} cl_vector;


typedef union {
    __bf16 f16;
    float f32;
    double f64;
} cl_number_t;

typedef struct {
    CL_TYPES type;
    cl_number_t value;
} cl_value_t;

typedef struct cl_dimensions_t {
    uint32_t dims_count;
    uint32_t* dims;
} cl_dimensions;

typedef struct  cl_index_t {
    uint32_t* indices;
    uint32_t count;
} cl_index;


typedef struct {
    uint32_t start;
    uint32_t end;
} cl_slice_el_t;

typedef struct {
    cl_slice_el_t* slices;
    uint32_t count;
} cl_slice_t;


typedef struct cl_tensor_t {
    void* data;
    cl_dimensions* dims;
    uint32_t size;
    CL_TYPES type;
} cl_tensor;


// data functions
cl_dimensions* cl_create_empty_dimensions(uint32_t dims_count, ccb_arena* arena);
cl_slice_t* cl_create_empty_slice(uint32_t count, ccb_arena* arena);
cl_index* cl_create_empty_index(uint32_t count, ccb_arena* arena);

cl_vector* cl_create_vector(uint32_t size, CL_TYPES type, ccb_arena* arena);
cl_dimensions* cl_create_dimensions(uint32_t dims_count, ccb_arena* arena, uint32_t* dims);
cl_index* cl_create_index(uint32_t count, ccb_arena* arena, uint32_t* indices);
cl_slice_t* cl_create_slice(uint32_t count, ccb_arena* arena, uint32_t* starts, uint32_t* ends);
cl_tensor* cl_create_tensor(cl_dimensions* dims, CL_TYPES type, ccb_arena* arena);

cl_vector* cl_clone_vector(cl_vector* vector, ccb_arena* arena);
cl_tensor* cl_clone_tensor(cl_tensor* tensor, ccb_arena* arena);
cl_index* cl_clone_index(cl_index* index, ccb_arena* arena);
cl_slice_t* cl_clone_slice(cl_slice_t* slice, ccb_arena* arena);
cl_dimensions* cl_clone_dimensions(cl_dimensions* dimensions, ccb_arena* arena);


cl_value_t to_cl_value(double number, CL_TYPES);
__bf16 cl_value_to_f16(cl_value_t value);
float cl_value_to_f32(cl_value_t value);
double cl_value_to_f64(cl_value_t value);
cl_value_t cl_value_as(cl_value_t a, CL_TYPES target_type);


void cl_data_to_vector(cl_vector* vector, void* data, uint32_t count);
void cl_data_to_tensor(cl_tensor* tensor, void* data, uint32_t count);

void cl_print_vector(cl_vector* vector);
void cl_print_tensor(cl_tensor* tensor, ccb_arena* tmp_arena);

cl_value_t cl_get_vector_element(cl_vector* vector, uint32_t index);
void cl_set_vector_element(cl_vector* vector, uint32_t index, cl_value_t value);

cl_value_t cl_get_tensor_element(cl_tensor* tensor, cl_index* index);
cl_tensor* cl_get_sub_tensor(cl_tensor* tensor, cl_index* index, ccb_arena* arena);
void cl_set_tensor_element(cl_tensor* tensor, cl_index* index, cl_value_t value);


#endif // __DATA_H__