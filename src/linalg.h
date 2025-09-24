#ifndef __LINALG_H__
#define __LINALG_H__

#include <stdint.h>
#include "../ccbase/utils/mem.h"
#include "data.h"

// scalar maths

/* 
   type agnostic addition for cl_values_t
   !! a.type must equal b.type
*/
cl_value_t cl_scalar_add(cl_value_t a, cl_value_t b);
/*
    type agnostic substraction for cl_values_t
   !! a.type must equal b.type
*/
cl_value_t cl_scalar_sub(cl_value_t a, cl_value_t b);
/*
    type agnostic multiplication for cl_values_t
   !! a.type must equal b.type
*/
cl_value_t cl_scalar_mul(cl_value_t a, cl_value_t b);
/*
    type agnostic division for cl_values_t
   !! a.type must equal b.type
*/
cl_value_t cl_scalar_div(cl_value_t a, cl_value_t b);
/*
   type agnostic abs
*/
cl_value_t cl_scalar_abs(cl_value_t a);
/*
    type agnostic power for cl_values_t
   !! a.type must equal b.type
*/
cl_value_t cl_scalar_pow(cl_value_t a, cl_value_t b);
/*
    type agnostic root cl_values_t
   !! a.type must equal b.type
*/
cl_value_t cl_scalar_root(cl_value_t a, cl_value_t b);

/* 
   type agnostic addition for cl_values_t
   !! a.type must equal b.type
*/
cl_value_t cl_scalar_add_args(cl_value_t a, void* b);
/*
    type agnostic substraction for cl_values_t
   !! a.type must equal b.type
*/
cl_value_t cl_scalar_sub_args(cl_value_t a, void* b);
/*
    type agnostic multiplication for cl_values_t
   !! a.type must equal b.type
*/
cl_value_t cl_scalar_mul_args(cl_value_t a, void* b);
/*
    type agnostic division for cl_values_t
   !! a.type must equal b.type
*/
cl_value_t cl_scalar_div_args(cl_value_t a, void* b);
/*
    type agnostic power for cl_values_t
   !! a.type must equal b.type
*/
cl_value_t cl_scalar_pow_args(cl_value_t a, void* b);
/*
    type agnostic root cl_values_t
   !! a.type must equal b.type
*/
cl_value_t cl_scalar_root_args(cl_value_t a, void* b);


// vector operations

/* Element-wise addition between 2 cl_vector
   - cl_vector* a: first input vector
   - cl_vector* b: second input vector
   - ccb_arena* arena: arena where the result vector will be allocated
   - return: a pointer to the result vector
*/
cl_vector* cl_vector_add(cl_vector* a, cl_vector* b, ccb_arena* arena);
/* Element-wise addition between 2 cl_vector
   - cl_vector* a: first input vector
   - cl_vector* b: second input vector
   - return: a pointer to the result vector (a)
    !! the value in the 1st vector will be replaced by the results

*/
cl_vector* cl_vector_add_inplace(cl_vector* a, cl_vector* b);
/* Scalar addition between a cl_vector and a scalar
   - cl_vector* a: input vector
   - cl_value_t b: scalar
   - ccb_arena* arena: arena where the result vector will be allocated
   - return: a pointer to the result vector
*/
cl_vector* cl_vector_add_scalar(cl_vector* a, cl_value_t b, ccb_arena* arena);
/* Scalar addition between a cl_vector and a scalar
   - cl_vector* a: input vector
   - cl_value_t b: scalar
   - return: a pointer to the result vector (a)
   !! the value in the 1st vector will be replaced by the results
*/
cl_vector* cl_vector_add_scalar_inplace(cl_vector* a, cl_value_t b);

/* Element-wise subtraction between 2 cl_vector
   - cl_vector* a: first input vector
   - cl_vector* b: second input vector
   - ccb_arena* arena: arena where the result vector will be allocated
   - return: a pointer to the result vector
*/
cl_vector* cl_vector_sub(cl_vector* a, cl_vector* b, ccb_arena* arena);
/* Element-wise subtraction between 2 cl_vector
   - cl_vector* a: first input vector
   - cl_vector* b: second input vector
   - return: a pointer to the result vector (a)
   !! the value in the 1st vector will be replaced by the results
*/
cl_vector* cl_vector_sub_inplace(cl_vector* a, cl_vector* b);
/* Scalar subtraction from a cl_vector by a scalar
   - cl_vector* a: input vector
   - cl_value_t b: scalar
   - ccb_arena* arena: arena where the result vector will be allocated
   - return: a pointer to the result vector
*/
cl_vector* cl_vector_sub_scalar(cl_vector* a, cl_value_t b, ccb_arena* arena);
/* Scalar subtraction from a cl_vector by a scalar
   - cl_vector* a: input vector
   - cl_value_t b: scalar
   - return: a pointer to the result vector (a)
   !! the value in the 1st vector will be replaced by the results
*/
cl_vector* cl_vector_sub_scalar_inplace(cl_vector* a, cl_value_t b);

/* Element-wise multiplication between 2 cl_vector
   - cl_vector* a: first input vector
   - cl_vector* b: second input vector
   - ccb_arena* arena: arena where the result vector will be allocated
   - return: a pointer to the result vector
*/
cl_vector* cl_vector_mul_ellement_wise(cl_vector* a, cl_vector* b, ccb_arena* arena);
/* Element-wise multiplication between 2 cl_vector
   - cl_vector* a: first input vector
   - cl_vector* b: second input vector
   - return: a pointer to the result vector (a)
   !! the value in the 1st vector will be replaced by the results
*/
cl_vector* cl_vector_mul_ellement_wise_inplace(cl_vector* a, cl_vector* b);
/* Scalar multiplication of a cl_vector by a scalar
   - cl_vector* a: input vector
   - cl_value_t b: scalar
   - ccb_arena* arena: arena where the result vector will be allocated
   - return: a pointer to the result vector
*/
cl_vector* cl_vector_mul_scalar(cl_vector* a, cl_value_t b, ccb_arena* arena);
/* Scalar multiplication of a cl_vector by a scalar
   - cl_vector* a: input vector
   - cl_value_t b: scalar
   - return: a pointer to the result vector (a)
   !! the value in the 1st vector will be replaced by the results
*/
cl_vector* cl_vector_mul_scalar_inplace(cl_vector* a, cl_value_t b);

/* Element-wise division between 2 cl_vector
   - cl_vector* a: first input vector (dividend)
   - cl_vector* b: second input vector (divisor)
   - ccb_arena* arena: arena where the result vector will be allocated
   - return: a pointer to the result vector
*/
cl_vector* cl_vector_div_ellement_wise(cl_vector* a, cl_vector* b, ccb_arena* arena);
/* Element-wise division between 2 cl_vector
   - cl_vector* a: first input vector (dividend)
   - cl_vector* b: second input vector (divisor)
   - return: a pointer to the result vector (a)
   !! the value in the 1st vector will be replaced by the results
*/
cl_vector* cl_vector_div_ellement_wise_inplace(cl_vector* a, cl_vector* b);
/* Scalar division of a cl_vector by a scalar
   - cl_vector* a: input vector (dividend)
   - cl_value_t b: scalar (divisor)
   - ccb_arena* arena: arena where the result vector will be allocated
   - return: a pointer to the result vector
*/
cl_vector* cl_vector_div_scalar(cl_vector* a, cl_value_t b, ccb_arena* arena);
/* Scalar division of a cl_vector by a scalar
   - cl_vector* a: input vector (dividend)
   - cl_value_t b: scalar (divisor)
   - return: a pointer to the result vector (a)
   !! the value in the 1st vector will be replaced by the results
*/
cl_vector* cl_vector_div_scalar_inplace(cl_vector* a, cl_value_t b);

/* Computes the dot product of two vectors.
   - cl_vector* a: first input vector
   - cl_vector* b: second input vector
   - return: a cl_value_t containing the dot product
*/
cl_value_t cl_vector_dot(cl_vector* a, cl_vector* b);
/* Computes the cross product of two 3D vectors.
   - cl_vector* a: first input vector (must be size 3)
   - cl_vector* b: second input vector (must be size 3)
   - ccb_arena* arena: arena where the result vector will be allocated
   - return: a pointer to the result vector
*/
cl_vector* cl_vector_cross(cl_vector* a, cl_vector* b, ccb_arena* arena);
/* Computes the cross product of two 3D vectors, in-place.
   - cl_vector* a: first input vector (must be size 3)
   - cl_vector* b: second input vector (must be size 3)
   - return: a pointer to the result vector (a)
   !! the value in the 1st vector will be replaced by the results
*/
cl_vector* cl_vector_cross_inplace(cl_vector* a, cl_vector* b);

/* Computes the Euclidean norm (length) of a vector.
   - cl_vector* a: input vector
   - return: a cl_value_t containing the norm
*/
cl_value_t cl_vector_norm(cl_vector* a, uint32_t p, ccb_arena* tmp_arena);
/* Normalizes a vector to have a length of 1.
   - cl_vector* a: input vector
   - uint32_t p: power of the norm 
   - ccb_arena* tmp_arena: arena where a tmp vector will be allocated
   - return: the norm value od the vector
*/
cl_vector* cl_vector_normalize(cl_vector* a, ccb_arena* arena);
/* Normalizes a vector to have a length of 1, in-place.
   - cl_vector* a: input vector
   - return: a pointer to the normalized vector (a)
   !! the value in the vector will be replaced by the results
*/
cl_vector* cl_vector_normalize_inplace(cl_vector* a, ccb_arena* test_arena);

/* Applies a function to each element of a vector, creating a new vector.
   - cl_vector* a: input vector
   - cl_value_t (*func)(cl_value_t): function to apply to each element
   - ccb_arena* arena: arena where the result vector will be allocated
   - return: a pointer to the new vector with the results
*/
cl_vector* cl_vector_map(cl_vector* a, cl_value_t (*func)(cl_value_t), ccb_arena* arena);
/* Applies a function to each element of a vector, in-place.
   - cl_vector* a: input vector
   - cl_value_t (*func)(cl_value_t): function to apply to each element
   - return: a pointer to the modified vector (a)
*/
cl_vector* cl_vector_map_inplace(cl_vector* a, cl_value_t (*func)(cl_value_t));
/* Reduces a vector to a single value by applying a function cumulatively.
   - cl_vector* a: input vector
   - cl_value_t (*func)(cl_value_t, cl_value_t): accumulator function
   - cl_value_t initial: the initial value for the accumulator
   - return: the final accumulated cl_value_t
*/
cl_vector* cl_vector_map_args(cl_vector* a, cl_value_t (*func)(cl_value_t, void*), ccb_arena* arena, void* args);
/* Applies a function to each element of a vector, in-place.
   - cl_vector* a: input vector
   - cl_value_t (*func)(cl_value_t, void*): function to apply to each element
   - void* args: args to the maped function
   - return: a pointer to the modified vector (a)
*/
cl_vector* cl_vector_map_args_inplace(cl_vector* a, cl_value_t (*func)(cl_value_t, void*), void* args);
/* Reduces a vector to a single value by applying a function cumulatively.
   - cl_vector* a: input vector
   - cl_value_t (*func)(cl_value_t, cl_value_t, void*): accumulator function
   - cl_value_t initial: the initial value for the accumulator
   - void* args: args to the maped function
   - return: the final accumulated cl_value_t
*/
cl_value_t cl_vector_reduce(cl_vector* a, cl_value_t (*func)(cl_value_t, cl_value_t), cl_value_t initial);
/* Generic element-wise operation between two vectors, creating a new vector.
   - cl_vector* a: first input vector
   - cl_vector* b: second input vector
   - cl_value_t (*func)(cl_value_t, cl_value_t): function for the element-wise operation
   - ccb_arena* arena: arena where the result vector will be allocated
   - return: a pointer to the new vector
*/
cl_vector* cl_for_each_vector_op(cl_vector* a, cl_vector* b, cl_value_t (*func)(cl_value_t, cl_value_t), ccb_arena* arena);
/* Generic in-place element-wise operation between two vectors.
   - cl_vector* a: first input vector
   - cl_vector* b: second input vector
   - cl_value_t (*func)(cl_value_t, cl_value_t): function for the element-wise operation
   - return: a pointer to the modified vector (a)
   !! the value in the 1st vector will be replaced by the results
*/
cl_vector* cl_for_each_vector_op_inplace(cl_vector* a, cl_vector* b, cl_value_t (*func)(cl_value_t, cl_value_t));
/* Generic element-wise operation between a vector and a scalar, creating a new vector.
   - cl_vector* a: input vector
   - cl_value_t b: scalar value
   - cl_value_t (*func)(cl_value_t, cl_value_t): function for the element-wise operation
   - ccb_arena* arena: arena where the result vector will be allocated
   - return: a pointer to the new vector
*/
cl_vector* cl_for_each_vector_scalar_op(cl_vector* a, cl_value_t b, cl_value_t (*func)(cl_value_t, cl_value_t), ccb_arena* arena);
/* Generic in-place element-wise operation between a vector and a scalar.
   - cl_vector* a: input vector
   - cl_value_t b: scalar value
   - cl_value_t (*func)(cl_value_t, cl_value_t): function for the element-wise operation
   - return: a pointer to the modified vector (a)
   !! the value in the vector will be replaced by the results
*/
cl_vector* cl_for_each_vector_scalar_op_inplace(cl_vector* a, cl_value_t b, cl_value_t (*func)(cl_value_t, cl_value_t));


#endif