# Scandium
A high performance machine learning library written in c 

This project is a work in progress and something I do alone in my free time,
there may be bugs and it may not work on your specific machine/OS

## Usage
code:
```c
#include "scandium.h"

int main(){
    ccb_InitLog("logs.log");
    ccb_arena* arena = ccb_init_arena();
    
    float a[10] = {1,  2, 3, 4, 5, 6, 7, 8, 9, 10};
    float b[10] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

    sc_vector* vec1 = sc_create_vector(size, sc_float32, arena);
    sc_vector* vec2 = sc_create_vector(size, sc_float32, arena);
    
    sc_data_to_vector(vec1, a, 10);
    sc_data_to_vector(vec2, b, 10);

    sc_vector* result = c_vector_add(vec1, vec2, arena);
    sc_print_vector(result);
    
    ccb_arena_free(arena);
    ccb_CloseLogFile();

    return 0;
}
```

compilation
```bash
$ gcc main.c ./build/scandium.a -o main
```


## Elements
- linalg: a linear algebra library for tensors and vectors
- scandium engine: a execution engine supporting multi threading, SIMD instructions, and batch operations

## WIP
- implement tensor operations
- implement SIMD supports
- migrate all vector operations to the scandium engine
- improve the documentation and api

## TODO
- implement linear regression
- implement logistic regression
- implement neural networks with a modular architecture