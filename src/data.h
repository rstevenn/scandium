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
    sc_float16,
    sc_float32,
    sc_float64,
} sc_TYPES;


typedef struct sc_vector_t {
    void* data;
    uint32_t size;
    sc_TYPES type;
} sc_vector;


typedef union {
    __bf16 f16;
    float f32;
    double f64;
} sc_number_t;

typedef struct {
    sc_TYPES type;
    sc_number_t value;
} sc_value_t;

typedef struct sc_dimensions_t {
    uint32_t dims_count;
    uint32_t* dims;
} sc_dimensions;

typedef struct  sc_index_t {
    uint32_t* indices;
    uint32_t count;
} sc_index;


typedef struct {
    uint32_t start;
    uint32_t end;
} sc_slice_el_t;

typedef struct {
    sc_slice_el_t* slices;
    uint32_t count;
} sc_slice_t;


typedef struct sc_tensor_t {
    void* data;
    sc_dimensions* dims;
    uint32_t size;
    sc_TYPES type;
} sc_tensor;


// data functions
sc_dimensions* sc_create_empty_dimensions(uint32_t dims_count, ccb_arena* arena);
sc_slice_t* sc_create_empty_slice(uint32_t count, ccb_arena* arena);
sc_index* sc_create_empty_index(uint32_t count, ccb_arena* arena);

sc_vector* sc_create_vector(uint32_t size, sc_TYPES type, ccb_arena* arena);
sc_dimensions* sc_create_dimensions(uint32_t dims_count, ccb_arena* arena, uint32_t* dims);
sc_index* sc_create_index(uint32_t count, ccb_arena* arena, uint32_t* indices);
sc_slice_t* sc_create_slice(uint32_t count, ccb_arena* arena, uint32_t* starts, uint32_t* ends);
sc_tensor* sc_create_tensor(sc_dimensions* dims, sc_TYPES type, ccb_arena* arena);

sc_vector* sc_clone_vector(sc_vector* vector, ccb_arena* arena);
sc_tensor* sc_clone_tensor(sc_tensor* tensor, ccb_arena* arena);
sc_index* sc_clone_index(sc_index* index, ccb_arena* arena);
sc_slice_t* sc_clone_slice(sc_slice_t* slice, ccb_arena* arena);
sc_dimensions* sc_clone_dimensions(sc_dimensions* dimensions, ccb_arena* arena);


sc_value_t to_sc_value(double number, sc_TYPES);
__bf16 sc_value_to_f16(sc_value_t value);
float sc_value_to_f32(sc_value_t value);
double sc_value_to_f64(sc_value_t value);
sc_value_t sc_value_as(sc_value_t a, sc_TYPES target_type);


void sc_data_to_vector(sc_vector* vector, void* data, uint32_t count);
void sc_data_to_tensor(sc_tensor* tensor, void* data, uint32_t count);

void sc_print_vector(sc_vector* vector);
void sc_print_tensor(sc_tensor* tensor, ccb_arena* tmp_arena);

sc_value_t sc_get_vector_element(sc_vector* vector, uint32_t index);
void sc_set_vector_element(sc_vector* vector, uint32_t index, sc_value_t value);

sc_value_t sc_get_tensor_element(sc_tensor* tensor, sc_index* index);
sc_tensor* sc_get_sub_tensor(sc_tensor* tensor, sc_index* index, ccb_arena* arena);
void sc_set_tensor_element(sc_tensor* tensor, sc_index* index, sc_value_t value);


#endif // __DATA_H__