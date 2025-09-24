#ifndef __MEM_H__
#define __MEM_H__

#include <stdlib.h>
#include <stdint.h>

#define byte sizeof(char)
#define kb 1024*byte
#define mb 1024*kb
#define gb 1024*mb
#define tb 1024*gb

/*
    Arena allocator
*/

typedef struct _ccb_arena_type {
    unsigned char* data;

    uint64_t capacity;

    struct _ccb_arena_type* next;
} ccb_arena;


// custom malloc/free version
ccb_arena* ccb_init_arena(void);
void* ccb_arena_malloc(ccb_arena* arena, uint64_t size);

void ccb_arena_reset(ccb_arena* arena);
void ccb_arena_free(ccb_arena* arena);

// no os version
void ccb_arena_nos_setup_memory(unsigned char* ram, uint64_t ram_size);
void ccb_arena_nos_reset_ram(unsigned char* ram);

ccb_arena* ccb_init_nos_arena(unsigned char* ram);
void* ccb_nos_arena_malloc(unsigned char* ram, ccb_arena* arena, uint64_t size);

void ccb_nos_arena_reset(ccb_arena* arena);
void ccb_nos_arena_free(unsigned char* ram, ccb_arena* arena);


// const
#ifndef CCB_ARENA_CAPACITY
    #define CCB_ARENA_CAPACITY gb
#endif

#ifndef CCB_ARENA_MALLOC
    #define CCB_ARENA_MALLOC malloc
#endif

#ifndef CCB_ARENA_FREE
    #define CCB_ARENA_FREE free
#endif

// Implementation
#ifdef CCB_ARENA_IMPL
#ifndef CCB_LOGLEVEL
    #define CCB_LOGLEVEL 2
#endif

#include "../logs/log.h"

// USING CCB_ARENA_MALLOC CCB_ARENA_MALLOC
ccb_arena* ccb_init_arena(void) {
    ccb_arena* arena = (ccb_arena*)CCB_ARENA_MALLOC(sizeof(ccb_arena));
    CCB_NOTNULL(arena, "can't allocate memory for a new memory block metadat")

    arena->data = (unsigned char*)CCB_ARENA_MALLOC(CCB_ARENA_CAPACITY);
    arena->capacity = CCB_ARENA_CAPACITY;
    CCB_NOTNULL(arena->data, "can't allocate a new memory block")
    arena->next = NULL;

    return arena;
}


void* ccb_arena_malloc(ccb_arena* arena, uint64_t size) {
    if (size > CCB_ARENA_CAPACITY)
        return NULL;

    ccb_arena* current_arena = arena;
    
    while (current_arena->capacity < size || current_arena->capacity < 1) {
        if (current_arena->next == NULL) {
            current_arena->next = ccb_init_arena();
        }
        current_arena = current_arena->next;
    }
    
    uint64_t allocated_offset = CCB_ARENA_CAPACITY - current_arena->capacity;
    current_arena->capacity -= size;
    return (void*) (current_arena->data + allocated_offset);
}


void ccb_arena_reset(ccb_arena* arena) {
    ccb_arena* current_arena = arena;

    do {
        arena->capacity = CCB_ARENA_CAPACITY;
        current_arena = current_arena->next;
    } while (current_arena != NULL);
}


void ccb_arena_free(ccb_arena* arena) {
    ccb_arena* current_arena = arena;
    ccb_arena* next_arena;

    do {
        next_arena = current_arena->next;
        CCB_ARENA_FREE(current_arena->data);
        CCB_ARENA_FREE(current_arena);
        current_arena = next_arena;
    } while (current_arena != NULL);
}


// NO OS dependent version
typedef struct _ccb_arena_ram_data {
    void* blocks_status;
    void* blocks;
    uint64_t max_block_numbers;
    uint64_t ram_size;
} ccb_arena_ram_data;


void ccb_arena_nos_setup_memory(unsigned char* ram, uint64_t ram_size) {

    /* ram structure
        [ccb_area_ram_data, table of allocated blocks, BLock 1, ..., BLock max_block_numbers]
        BLock = <ccb_arena, area_data>
    */
    ccb_arena_ram_data data;
    data.ram_size = ram_size;

    // calculate nb of blocks and adresses
    data.max_block_numbers = (ram_size - sizeof(ccb_arena_ram_data)) / (1 + CCB_ARENA_CAPACITY + sizeof(ccb_arena));
    CCB_CHECK(data.max_block_numbers > 0, "not enough space for block allocation")
    data.blocks_status = ram + sizeof(ccb_arena_ram_data);
    data.blocks = (void*)((unsigned char*)data.blocks_status + data.max_block_numbers);

    // write to ram
    ((ccb_arena_ram_data*)ram)[0] = data;

    // int the block_status
    ccb_arena_nos_reset_ram(ram);
}


void ccb_arena_nos_reset_ram(unsigned char* ram) {
    ccb_arena_ram_data meta_data = ((ccb_arena_ram_data*)ram)[0];

    // write all 0 on the blocks status table
    unsigned char* status_index = (unsigned char*)meta_data.blocks_status;
    for (size_t i = 0; i < meta_data.max_block_numbers; i++) {
        *status_index = 0;
        status_index++;
    }
}


ccb_arena* ccb_init_nos_arena(unsigned char* ram) {
    ccb_arena_ram_data meta_data = ((ccb_arena_ram_data*)ram)[0];

    // found a free block
    unsigned char* status_index = (unsigned char*) meta_data.blocks_status;
    size_t block_index = 0;

    for (block_index = 0; block_index < meta_data.max_block_numbers; block_index++) {
        if (*status_index == 0) {
            break;
        }
        status_index++;
    }

    if (block_index == meta_data.max_block_numbers) {
        CCB_WARNING("No memory available to allocate")
        return NULL;
    }

    // write data
    ccb_arena* arena = (ccb_arena*) ((size_t)meta_data.blocks +  block_index*(CCB_ARENA_CAPACITY+sizeof(ccb_arena))); 

    arena->capacity = CCB_ARENA_CAPACITY;
    arena->next = NULL;
    arena->data = (unsigned char*)((size_t)arena + sizeof(ccb_arena));

    // update the status 
    *status_index = 1;

    return arena;
}


void* ccb_nos_arena_malloc(unsigned char* ram, ccb_arena* arena, uint64_t size) {

    if (size > CCB_ARENA_CAPACITY)
        return NULL;

    ccb_arena* current_arena = arena;
    
    while (current_arena->capacity < size || current_arena->capacity < 1) {
        if (current_arena->next == NULL) {
            current_arena->next = ccb_init_nos_arena(ram);
        }
        current_arena = current_arena->next;
    }
    
    uint64_t allocated_offset = CCB_ARENA_CAPACITY - current_arena->capacity;
    current_arena->capacity -= size;
    return (void*) (current_arena->data + allocated_offset);
}


void ccb_nos_arena_reset(ccb_arena* arena) {
    ccb_arena* current_arena = arena;

    do {
        arena->capacity = CCB_ARENA_CAPACITY;
        current_arena = current_arena->next;
    } while (current_arena != NULL);
}


void ccb_nos_arena_free(unsigned char* ram, ccb_arena* arena) {
    ccb_arena_ram_data meta_data = ((ccb_arena_ram_data*)ram)[0];
    ccb_arena* current_arena = arena;

    do {
        size_t block_id = ((size_t)current_arena - (size_t)meta_data.blocks) / (CCB_ARENA_CAPACITY + sizeof(ccb_arena));
        unsigned char* block_status_addr = (unsigned char*)(meta_data.blocks_status) + block_id; 
        *block_status_addr = 0;

        current_arena = current_arena->next;
    } while (current_arena != NULL);
}


#endif
#endif