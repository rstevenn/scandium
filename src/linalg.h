#ifndef __LINALG_H__
#define __LINALG_H__

#include <stdint.h>
#include "../ccbase/utils/mem.h"
#include "data.h"

// scalar maths

/* 
   type agnostic addition for sc_values_t
   !! a.type must equal b.type
*/
sc_value_t sc_scalar_add(sc_value_t a, sc_value_t b);
/*
    type agnostic substraction for sc_values_t
   !! a.type must equal b.type
*/
sc_value_t sc_scalar_sub(sc_value_t a, sc_value_t b);
/*
    type agnostic multiplication for sc_values_t
   !! a.type must equal b.type
*/
sc_value_t sc_scalar_mul(sc_value_t a, sc_value_t b);
/*
    type agnostic division for sc_values_t
   !! a.type must equal b.type
*/
sc_value_t sc_scalar_div(sc_value_t a, sc_value_t b);
/*
   type agnostic abs
*/
sc_value_t sc_scalar_abs(sc_value_t a);
/*
    type agnostic power for sc_values_t
   !! a.type must equal b.type
*/
sc_value_t sc_scalar_pow(sc_value_t a, sc_value_t b);
/*
    type agnostic root sc_values_t
   !! a.type must equal b.type
*/
sc_value_t sc_scalar_root(sc_value_t a, sc_value_t b);

/* 
   type agnostic addition for sc_values_t
   !! a.type must equal b.type
*/
sc_value_t sc_scalar_add_args(sc_value_t a, void* b);
/*
    type agnostic substraction for sc_values_t
   !! a.type must equal b.type
*/
sc_value_t sc_scalar_sub_args(sc_value_t a, void* b);
/*
    type agnostic multiplication for sc_values_t
   !! a.type must equal b.type
*/
sc_value_t sc_scalar_mul_args(sc_value_t a, void* b);
/*
    type agnostic division for sc_values_t
   !! a.type must equal b.type
*/
sc_value_t sc_scalar_div_args(sc_value_t a, void* b);
/*
    type agnostic power for sc_values_t
   !! a.type must equal b.type
*/
sc_value_t sc_scalar_pow_args(sc_value_t a, void* b);
/*
    type agnostic root sc_values_t
   !! a.type must equal b.type
*/
sc_value_t sc_scalar_root_args(sc_value_t a, void* b);


// vector operations

/* Element-wise addition between 2 sc_vector
   - sc_vector* a: first input vector
   - sc_vector* b: second input vector
   - ccb_arena* arena: arena where the result vector will be allocated
   - return: a pointer to the result vector
*/
sc_vector* sc_vector_add(sc_vector* a, sc_vector* b, ccb_arena* arena);
/* Element-wise addition between 2 sc_vector
   - sc_vector* a: first input vector
   - sc_vector* b: second input vector
   - return: a pointer to the result vector (a)
    !! the value in the 1st vector will be replaced by the results

*/
sc_vector* sc_vector_add_inplace(sc_vector* a, sc_vector* b);
/* Scalar addition between a sc_vector and a scalar
   - sc_vector* a: input vector
   - sc_value_t b: scalar
   - ccb_arena* arena: arena where the result vector will be allocated
   - return: a pointer to the result vector
*/
sc_vector* sc_vector_add_scalar(sc_vector* a, sc_value_t b, ccb_arena* arena);
/* Scalar addition between a sc_vector and a scalar
   - sc_vector* a: input vector
   - sc_value_t b: scalar
   - return: a pointer to the result vector (a)
   !! the value in the 1st vector will be replaced by the results
*/
sc_vector* sc_vector_add_scalar_inplace(sc_vector* a, sc_value_t b);

/* Element-wise subtraction between 2 sc_vector
   - sc_vector* a: first input vector
   - sc_vector* b: second input vector
   - ccb_arena* arena: arena where the result vector will be allocated
   - return: a pointer to the result vector
*/
sc_vector* sc_vector_sub(sc_vector* a, sc_vector* b, ccb_arena* arena);
/* Element-wise subtraction between 2 sc_vector
   - sc_vector* a: first input vector
   - sc_vector* b: second input vector
   - return: a pointer to the result vector (a)
   !! the value in the 1st vector will be replaced by the results
*/
sc_vector* sc_vector_sub_inplace(sc_vector* a, sc_vector* b);
/* Scalar subtraction from a sc_vector by a scalar
   - sc_vector* a: input vector
   - sc_value_t b: scalar
   - ccb_arena* arena: arena where the result vector will be allocated
   - return: a pointer to the result vector
*/
sc_vector* sc_vector_sub_scalar(sc_vector* a, sc_value_t b, ccb_arena* arena);
/* Scalar subtraction from a sc_vector by a scalar
   - sc_vector* a: input vector
   - sc_value_t b: scalar
   - return: a pointer to the result vector (a)
   !! the value in the 1st vector will be replaced by the results
*/
sc_vector* sc_vector_sub_scalar_inplace(sc_vector* a, sc_value_t b);

/* Element-wise multiplication between 2 sc_vector
   - sc_vector* a: first input vector
   - sc_vector* b: second input vector
   - ccb_arena* arena: arena where the result vector will be allocated
   - return: a pointer to the result vector
*/
sc_vector* sc_vector_mul_ellement_wise(sc_vector* a, sc_vector* b, ccb_arena* arena);
/* Element-wise multiplication between 2 sc_vector
   - sc_vector* a: first input vector
   - sc_vector* b: second input vector
   - return: a pointer to the result vector (a)
   !! the value in the 1st vector will be replaced by the results
*/
sc_vector* sc_vector_mul_ellement_wise_inplace(sc_vector* a, sc_vector* b);
/* Scalar multiplication of a sc_vector by a scalar
   - sc_vector* a: input vector
   - sc_value_t b: scalar
   - ccb_arena* arena: arena where the result vector will be allocated
   - return: a pointer to the result vector
*/
sc_vector* sc_vector_mul_scalar(sc_vector* a, sc_value_t b, ccb_arena* arena);
/* Scalar multiplication of a sc_vector by a scalar
   - sc_vector* a: input vector
   - sc_value_t b: scalar
   - return: a pointer to the result vector (a)
   !! the value in the 1st vector will be replaced by the results
*/
sc_vector* sc_vector_mul_scalar_inplace(sc_vector* a, sc_value_t b);

/* Element-wise division between 2 sc_vector
   - sc_vector* a: first input vector (dividend)
   - sc_vector* b: second input vector (divisor)
   - ccb_arena* arena: arena where the result vector will be allocated
   - return: a pointer to the result vector
*/
sc_vector* sc_vector_div_ellement_wise(sc_vector* a, sc_vector* b, ccb_arena* arena);
/* Element-wise division between 2 sc_vector
   - sc_vector* a: first input vector (dividend)
   - sc_vector* b: second input vector (divisor)
   - return: a pointer to the result vector (a)
   !! the value in the 1st vector will be replaced by the results
*/
sc_vector* sc_vector_div_ellement_wise_inplace(sc_vector* a, sc_vector* b);
/* Scalar division of a sc_vector by a scalar
   - sc_vector* a: input vector (dividend)
   - sc_value_t b: scalar (divisor)
   - ccb_arena* arena: arena where the result vector will be allocated
   - return: a pointer to the result vector
*/
sc_vector* sc_vector_div_scalar(sc_vector* a, sc_value_t b, ccb_arena* arena);
/* Scalar division of a sc_vector by a scalar
   - sc_vector* a: input vector (dividend)
   - sc_value_t b: scalar (divisor)
   - return: a pointer to the result vector (a)
   !! the value in the 1st vector will be replaced by the results
*/
sc_vector* sc_vector_div_scalar_inplace(sc_vector* a, sc_value_t b);

/* Computes the dot product of two vectors.
   - sc_vector* a: first input vector
   - sc_vector* b: second input vector
   - return: a sc_value_t containing the dot product
*/
sc_value_t sc_vector_dot(sc_vector* a, sc_vector* b);
/* Computes the cross product of two 3D vectors.
   - sc_vector* a: first input vector (must be size 3)
   - sc_vector* b: second input vector (must be size 3)
   - ccb_arena* arena: arena where the result vector will be allocated
   - return: a pointer to the result vector
*/
sc_vector* sc_vector_cross(sc_vector* a, sc_vector* b, ccb_arena* arena);
/* Computes the cross product of two 3D vectors, in-place.
   - sc_vector* a: first input vector (must be size 3)
   - sc_vector* b: second input vector (must be size 3)
   - return: a pointer to the result vector (a)
   !! the value in the 1st vector will be replaced by the results
*/
sc_vector* sc_vector_cross_inplace(sc_vector* a, sc_vector* b);

/* Computes the Euclidean norm (length) of a vector.
   - sc_vector* a: input vector
   - return: a sc_value_t containing the norm
*/
sc_value_t sc_vector_norm(sc_vector* a, uint64_t p, ccb_arena* tmp_arena);
/* Normalizes a vector to have a length of 1.
   - sc_vector* a: input vector
   - uint64_t p: power of the norm 
   - ccb_arena* tmp_arena: arena where a tmp vector will be allocated
   - return: the norm value od the vector
*/
sc_vector* sc_vector_normalize(sc_vector* a, ccb_arena* arena);
/* Normalizes a vector to have a length of 1, in-place.
   - sc_vector* a: input vector
   - return: a pointer to the normalized vector (a)
   !! the value in the vector will be replaced by the results
*/
sc_vector* sc_vector_normalize_inplace(sc_vector* a, ccb_arena* test_arena);

/* Applies a function to each element of a vector, creating a new vector.
   - sc_vector* a: input vector
   - sc_value_t (*func)(sc_value_t): function to apply to each element
   - ccb_arena* arena: arena where the result vector will be allocated
   - return: a pointer to the new vector with the results
*/
sc_vector* sc_vector_map(sc_vector* a, sc_value_t (*func)(sc_value_t), ccb_arena* arena);
/* Applies a function to each element of a vector, in-place.
   - sc_vector* a: input vector
   - sc_value_t (*func)(sc_value_t): function to apply to each element
   - return: a pointer to the modified vector (a)
*/
sc_vector* sc_vector_map_inplace(sc_vector* a, sc_value_t (*func)(sc_value_t));
/* Reduces a vector to a single value by applying a function cumulatively.
   - sc_vector* a: input vector
   - sc_value_t (*func)(sc_value_t, sc_value_t): accumulator function
   - sc_value_t initial: the initial value for the accumulator
   - return: the final accumulated sc_value_t
*/
sc_vector* sc_vector_map_args(sc_vector* a, sc_value_t (*func)(sc_value_t, void*), ccb_arena* arena, void* args);
/* Applies a function to each element of a vector, in-place.
   - sc_vector* a: input vector
   - sc_value_t (*func)(sc_value_t, void*): function to apply to each element
   - void* args: args to the maped function
   - return: a pointer to the modified vector (a)
*/
sc_vector* sc_vector_map_args_inplace(sc_vector* a, sc_value_t (*func)(sc_value_t, void*), void* args);
/* Reduces a vector to a single value by applying a function cumulatively.
   - sc_vector* a: input vector
   - sc_value_t (*func)(sc_value_t, sc_value_t, void*): accumulator function
   - sc_value_t initial: the initial value for the accumulator
   - void* args: args to the maped function
   - return: the final accumulated sc_value_t
*/
sc_value_t sc_vector_reduce(sc_vector* a, sc_value_t (*func)(sc_value_t, sc_value_t), sc_value_t initial);
/* Generic element-wise operation between two vectors, creating a new vector.
   - sc_vector* a: first input vector
   - sc_vector* b: second input vector
   - sc_value_t (*func)(sc_value_t, sc_value_t): function for the element-wise operation
   - ccb_arena* arena: arena where the result vector will be allocated
   - return: a pointer to the new vector
*/
sc_vector* sc_for_each_vector_op(sc_vector* a, sc_vector* b, sc_value_t (*func)(sc_value_t, sc_value_t), ccb_arena* arena);
/* Generic in-place element-wise operation between two vectors.
   - sc_vector* a: first input vector
   - sc_vector* b: second input vector
   - sc_value_t (*func)(sc_value_t, sc_value_t): function for the element-wise operation
   - return: a pointer to the modified vector (a)
   !! the value in the 1st vector will be replaced by the results
*/
sc_vector* sc_for_each_vector_op_inplace(sc_vector* a, sc_vector* b, sc_value_t (*func)(sc_value_t, sc_value_t));
/* Generic element-wise operation between a vector and a scalar, creating a new vector.
   - sc_vector* a: input vector
   - sc_value_t b: scalar value
   - sc_value_t (*func)(sc_value_t, sc_value_t): function for the element-wise operation
   - ccb_arena* arena: arena where the result vector will be allocated
   - return: a pointer to the new vector
*/
sc_vector* sc_for_each_vector_scalar_op(sc_vector* a, sc_value_t b, sc_value_t (*func)(sc_value_t, sc_value_t), ccb_arena* arena);
/* Generic in-place element-wise operation between a vector and a scalar.
   - sc_vector* a: input vector
   - sc_value_t b: scalar value
   - sc_value_t (*func)(sc_value_t, sc_value_t): function for the element-wise operation
   - return: a pointer to the modified vector (a)
   !! the value in the vector will be replaced by the results
*/
sc_vector* sc_for_each_vector_scalar_op_inplace(sc_vector* a, sc_value_t b, sc_value_t (*func)(sc_value_t, sc_value_t));


#endif