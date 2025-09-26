#include "data.h"
#include "const.h"
#include "../ccbase/logs/log.h"

#include <stdlib.h>
#include <string.h>


char** sc_TYPES_NAMES = (char*[]) {
    "float16",
    "float32",
    "float64"
};



// Create data structures
sc_dimensions* sc_create_empty_dimensions(uint64_t dims_count, ccb_arena* arena) {
    sc_dimensions* dimensions = (sc_dimensions*)ccb_arena_malloc(arena, sizeof(sc_dimensions));
    CCB_NOTNULL(dimensions, "Failed to allocate memory for dimensions struct");

    dimensions->dims_count = dims_count;
    dimensions->dims = (uint64_t*)ccb_arena_malloc(arena, dims_count * sizeof(uint64_t));
    CCB_NOTNULL(dimensions->dims, "Failed to allocate memory for dimensions array");

    return dimensions;
}


sc_slice* sc_create_empty_slice(uint64_t count, ccb_arena* arena) {
    sc_slice* slice = (sc_slice*)ccb_arena_malloc(arena, sizeof(sc_slice));
    CCB_NOTNULL(slice, "Failed to allocate memory for slice struct");

    slice->count = count;
    slice->slices = (sc_slice_el_t*)ccb_arena_malloc(arena, count * sizeof(sc_slice_el_t));
    CCB_NOTNULL(slice->slices, "Failed to allocate memory for slice array");

    return slice;
}


sc_index* sc_create_empty_index(uint64_t count, ccb_arena* arena) {
    sc_index* index = (sc_index*)ccb_arena_malloc(arena, sizeof(sc_index));
    CCB_NOTNULL(index, "Failed to allocate memory for index struct");

    index->count = count;
    index->indices = (uint64_t*)ccb_arena_malloc(arena, count * sizeof(uint64_t));
    CCB_NOTNULL(index->indices, "Failed to allocate memory for index array");

    return index;
}



sc_vector* sc_create_vector(uint64_t size, sc_TYPES type, ccb_arena* arena) {
    sc_vector* vector = (sc_vector*)ccb_arena_malloc(arena, sizeof(sc_vector));
    CCB_NOTNULL(vector, "Failed to allocate memory for vector struct");

    vector->size = size;
    vector->type = type;

    size_t type_size;
    switch (type) {
        case sc_float16:
            type_size = 2;
            break;
        case sc_float32:
            type_size = 4;
            break;
        case sc_float64:
            type_size = 8;
            break;
        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", type);
            return NULL;
    }

    vector->data = ccb_arena_malloc(arena, size * type_size);
    CCB_NOTNULL(vector->data, "Failed to allocate memory for vector data");

    return vector;    
}



sc_dimensions* sc_create_dimensions(uint64_t dims_count, ccb_arena* arena, uint64_t* dims) {
    sc_dimensions* dimensions = (sc_dimensions*)ccb_arena_malloc(arena, sizeof(sc_dimensions));
    CCB_NOTNULL(dimensions, "Failed to allocate memory for dimensions struct");

    dimensions->dims_count = dims_count;
    dimensions->dims = (uint64_t*)ccb_arena_malloc(arena, dims_count * sizeof(uint64_t));
    CCB_NOTNULL(dimensions->dims, "Failed to allocate memory for dimensions array");

    for (uint64_t i = 0; i < dims_count; i++) {
        dimensions->dims[i] = dims[i];
    }

    return dimensions;
}


sc_index* sc_create_index(uint64_t count, ccb_arena* arena, uint64_t* indices) {
    sc_index* index = (sc_index*)ccb_arena_malloc(arena, sizeof(sc_index));
    CCB_NOTNULL(index, "Failed to allocate memory for index struct");

    index->count = count;
    index->indices = (uint64_t*)ccb_arena_malloc(arena, count * sizeof(uint64_t));
    CCB_NOTNULL(index->indices, "Failed to allocate memory for index array");

    for (uint64_t i = 0; i < count; i++) {
        index->indices[i] = indices[i];
    }

    return index;
}

sc_slice* sc_create_slice(uint64_t count, ccb_arena* arena, uint64_t* starts, uint64_t* ends) {
    CCB_NOTNULL(arena, "Invalid arena pointer");
    CCB_NOTNULL(starts, "Invalid starts pointer");
    CCB_NOTNULL(ends, "Invalid ends pointer");

    sc_slice* slice = (sc_slice*)ccb_arena_malloc(arena, sizeof(sc_slice));
    CCB_NOTNULL(slice, "Failed to allocate memory for slice struct");

    slice->count = count;
    slice->slices = (sc_slice_el_t*)ccb_arena_malloc(arena, count * sizeof(sc_slice_el_t));
    CCB_NOTNULL(slice->slices, "Failed to allocate memory for slice array");

    for (uint64_t i = 0; i < count; i++) {
        slice->slices[i].start = starts[i];
        slice->slices[i].end = ends[i];
    }

    return slice;
}


sc_tensor* sc_create_tensor(sc_dimensions* dims, sc_TYPES type, ccb_arena* arena) {
    sc_tensor* tensor = (sc_tensor*)ccb_arena_malloc(arena, sizeof(sc_tensor));
    CCB_NOTNULL(tensor, "Failed to allocate memory for tensor struct");

    tensor->size = 1;
    for (uint64_t i = 0; i < dims->dims_count; i++) {
        tensor->size *= dims->dims[i];
    }

    tensor->type = type;

    size_t type_size;
    switch (type) {
        case sc_float16:
            type_size = 2;
            break;
        case sc_float32:
            type_size = 4;
            break;
        case sc_float64:
            type_size = 8;
            break;
        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", type);
            return NULL;
    }
    tensor->dims = dims;

    tensor->data = ccb_arena_malloc(arena, tensor->size * type_size);
    CCB_NOTNULL(tensor->data, "Failed to allocate memory for tensor data");

    return tensor;    
}

// clone functions
sc_vector* sc_clone_vector(sc_vector* vector, ccb_arena* arena) {
    sc_vector* clone = (sc_vector*)ccb_arena_malloc(arena, sizeof(sc_vector));
    CCB_NOTNULL(clone, "Failed to allocate memory for clone struct");

    clone->size = vector->size;
    clone->type = vector->type;

    size_t type_size;
    switch (vector->type) {
        case sc_float16:
            type_size = 2;
            break;
        case sc_float32:
            type_size = 4;
            break;
        case sc_float64:
            type_size = 8;
            break;
        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", vector->type);
            return NULL;
    }

    clone->data = ccb_arena_malloc(arena, vector->size * type_size);
    CCB_NOTNULL(clone->data, "Failed to allocate memory for clone data");

    memcpy(clone->data, vector->data, vector->size * type_size);

    return clone;
}


sc_tensor* sc_clone_tensor(sc_tensor* tensor, ccb_arena* arena) {
    sc_tensor* clone = (sc_tensor*)ccb_arena_malloc(arena, sizeof(sc_tensor));
    CCB_NOTNULL(clone, "Failed to allocate memory for clone struct");

    clone->size = tensor->size;
    clone->type = tensor->type;
    clone->dims = sc_clone_dimensions(tensor->dims, arena);
    CCB_NOTNULL(clone->dims, "Failed to clone dimensions");

    size_t type_size;
    switch (tensor->type) {
        case sc_float16:
            type_size = 2;
            break;
        case sc_float32:
            type_size = 4;
            break;
        case sc_float64:
            type_size = 8;
            break;
        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", tensor->type);
            return NULL;
    }

    clone->data = ccb_arena_malloc(arena, tensor->size * type_size);
    CCB_NOTNULL(clone->data, "Failed to allocate memory for clone data");

    memcpy(clone->data, tensor->data, tensor->size * type_size);

    return clone;
}

sc_index* sc_clone_index(sc_index* index, ccb_arena* arena) {
    sc_index* clone = (sc_index*)ccb_arena_malloc(arena, sizeof(sc_index));
    CCB_NOTNULL(clone, "Failed to allocate memory for clone struct");

    clone->count = index->count;

    clone->indices = (uint64_t*)ccb_arena_malloc(arena, index->count * sizeof(uint64_t));
    CCB_NOTNULL(clone->indices, "Failed to allocate memory for clone indices");

    memcpy(clone->indices, index->indices, index->count * sizeof(uint64_t));

    return clone;
}

sc_slice* sc_clone_slice(sc_slice* slice, ccb_arena* arena) {
    sc_slice* clone = (sc_slice*)ccb_arena_malloc(arena, sizeof(sc_slice));
    CCB_NOTNULL(clone, "Failed to allocate memory for clone struct");

    clone->count = slice->count;

    clone->slices = (sc_slice_el_t*)ccb_arena_malloc(arena, slice->count * sizeof(sc_slice_el_t));
    CCB_NOTNULL(clone->slices, "Failed to allocate memory for clone slices");

    memcpy(clone->slices, slice->slices, slice->count * sizeof(sc_slice_el_t));

    return clone;
}


sc_dimensions* sc_clone_dimensions(sc_dimensions* dimensions, ccb_arena* arena) {
    sc_dimensions* clone = (sc_dimensions*)ccb_arena_malloc(arena, sizeof(sc_dimensions));
    CCB_NOTNULL(clone, "Failed to allocate memory for clone struct");

    clone->dims_count = dimensions->dims_count;

    clone->dims = (uint64_t*)ccb_arena_malloc(arena, dimensions->dims_count * sizeof(uint64_t));
    CCB_NOTNULL(clone->dims, "Failed to allocate memory for clone dims");

    memcpy(clone->dims, dimensions->dims, dimensions->dims_count * sizeof(uint64_t));

    return clone;
}

// Load data
void sc_data_to_vector(sc_vector* vector, void* data, uint64_t size) {
    size_t type_size;
    switch (vector->type) {
        case sc_float16:
            type_size = 2;
            break;
        case sc_float32:
            type_size = 4;
            break;
        case sc_float64:
            type_size = 8;
            break;
        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", vector->type);
            return;
    }

    if (size > vector->size) {
        CCB_WARNING("Data size exceeds vector capacity, truncating data");
        size = vector->size;

    } else if (size < vector->size) {
        CCB_WARNING("Data size is less than vector capacity, zeroing remaining space");

        memset((unsigned char*)vector->data + size, 0, vector->size * type_size - size);
    }

    memcpy(vector->data, data, size*type_size);
}


void sc_data_to_tensor(sc_tensor* tensor, void* data, uint64_t size) {
    size_t type_size;
    switch (tensor->type) {
        case sc_float16:
            type_size = 2;
            break;
        case sc_float32:
            type_size = 4;
            break;
        case sc_float64:
            type_size = 8;
            break;
        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", tensor->type);
            return;
    }

    if (size > tensor->size) {
        CCB_WARNING("Data size exceeds tensor capacity, truncating data");
        size = tensor->size * type_size;

    } else if (size < tensor->size) {
        CCB_WARNING("Data size is less than tensor capacity, zeroing remaining space");

        memset((unsigned char*)tensor->data + size, 0, tensor->size * type_size - size);
    }

    memcpy(tensor->data, data, size*type_size);
}



// Print functions
void sc_print_vector(sc_vector* vector) {
    size_t type_size;
    switch (vector->type) {
        case sc_float16:
            type_size = 2;
            break;
        case sc_float32:
            type_size = 4;
            break;
        case sc_float64:
            type_size = 8;
            break;
        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", vector->type);
            return;
    }

    printf("Vector (size: %u, type: %s):\n", vector->size, sc_TYPES_NAMES[vector->type]);
    for (uint64_t i = 0; i < vector->size; i++) {
        if (vector->type == sc_float32) {
            float* data = (float*)vector->data;
            printf("  [%u]: %f\n", i, data[i]);
        } else if (vector->type == sc_float64) {
            double* data = (double*)vector->data;
            printf("  [%u]: %lf\n", i, data[i]);
        } else if (vector->type == sc_float16) {
            __bf16* data = (__bf16*)vector->data;
            printf("  [%u]: %f\n", i, (float)data[i]);
        } else {
            printf("  [%u]: <unsupported type for printing>\n", i);
        }
    }
}


void sc_print_tensor(sc_tensor* tensor, ccb_arena* tmp_arena) {
    size_t type_size;
    switch (tensor->type) {
        case sc_float16:
            type_size = 2;
            break;
        case sc_float32:
            type_size = 4;
            break;
        case sc_float64:
            type_size = 8;
            break;
        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", tensor->type);
            return;
    }

    sc_index index;
    index.count = tensor->dims->dims_count;
    index.indices = (uint64_t*)ccb_arena_malloc(tmp_arena, tensor->dims->dims_count * sizeof(uint64_t));
    CCB_NOTNULL(index.indices, "Failed to allocate memory for index array");
    memset(index.indices, 0, tensor->size * sizeof(uint64_t));

    printf("tensor (size: %u, type: %s, dimensions: [", tensor->size, sc_TYPES_NAMES[tensor->type]);
    for (uint64_t i = 0; i < tensor->dims->dims_count; i++) {
        printf("%u", tensor->dims->dims[i]);
        if (i < tensor->dims->dims_count - 1) {
            printf(", ");
        }
        index.indices[i] = 0;
        index.count--;
        if (index.count == 0) {
            index.count = tensor->dims->dims_count;
        }
    }
    printf("]):\n");

    for (uint64_t i = 0; i < tensor->size; i++) {
        printf("  [");
        for (uint64_t j = 0; j < tensor->dims->dims_count; j++) {
            printf("%u", index.indices[j]);
            if (j < tensor->dims->dims_count - 1) {
                printf(", ");
            }
        }
        printf("]: ");

        if (tensor->type == sc_float32) {
            float* data = (float*)tensor->data;
            printf("%f\n", data[i]);
        } else if (tensor->type == sc_float64) {
            double* data = (double*)tensor->data;
            printf("%lf\n", data[i]);
        } else if (tensor->type == sc_float16) {
            __bf16* data = (__bf16*)tensor->data;
            printf("%lf\n", (float)data[i]);
        } else {
            printf("<unsupported type for printing>\n");
        }

        // Increment index
        for (int32_t j = tensor->dims->dims_count - 1; j >= 0; j--) {
            index.indices[j]++;
            if (index.indices[j] < tensor->dims->dims[j]) {
                break;
            } else if (j > 0) {
                index.indices[j] = 0;
            }
        }
    }
    
}


void sc_print_index(sc_index* index) {
    printf("Index (count: %u): [", index->count);
    for (uint64_t i = 0; i < index->count; i++) {
        printf("%u", index->indices[i]);
        if (i < index->count - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

void sc_print_dimensions(sc_dimensions* dims) {
    printf("Dimensions (count: %u): [", dims->dims_count);
    for (uint64_t i = 0; i < dims->dims_count; i++) {
        printf("%u", dims->dims[i]);
        if (i < dims->dims_count - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

void sc_print_slice(sc_slice* slice) {
    printf("Slice (count: %u): [", slice->count);
    for (uint64_t i = 0; i < slice->count; i++) {
        printf("[%u, %u)", slice->slices[i].start, slice->slices[i].end);
        if (i < slice->count - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}



// Values functions
sc_value_t to_sc_value(double value, sc_TYPES type) {
    sc_value_t scalar;
    scalar.type = type;
    switch (type) {
        case sc_float16:
            scalar.value.f16 = (__bf16)value;
            break;
        case sc_float32:
            scalar.value.f32 = (float)value;
            break;
        case sc_float64:
            scalar.value.f64 = value;
            break;
        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", type);
            scalar.type = -1; // Invalid type
            break;
    }
    return scalar;
}


__bf16 sc_value_to_f16(sc_value_t value) {
    switch (value.type) {
        case sc_float16:
            return value.value.f16;
        case sc_float32:
            return (__bf16)value.value.f32;
        case sc_float64:
            return (__bf16)value.value.f64;
        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", value.type);
            return (__bf16)0.0; // Default return value
    }
}


float sc_value_to_f32(sc_value_t value) {
    switch (value.type) {
        case sc_float16:
            return (float)value.value.f16;
        case sc_float32:
            return value.value.f32;
        case sc_float64:
            return (float)value.value.f64;
        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", value.type);
            return 0.0f; // Default return value
    }
}


double sc_value_to_f64(sc_value_t value) {
    switch (value.type) {
        case sc_float16:
            return (double)value.value.f16;
        case sc_float32:
            return (double)value.value.f32;
        case sc_float64:
            return value.value.f64;
        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", value.type);
            return 0.0; // Default return value
    }
}


sc_value_t sc_value_as(sc_value_t a, sc_TYPES target_type) {
    double value = sc_value_to_f64(a); // Convert to double first for precision
    return to_sc_value(value, target_type);
}


// geters and seters
sc_value_t sc_get_vector_element(sc_vector* vector, uint64_t index) {
    if (index >= vector->size) {
        CCB_ERROR("Index %u out of bounds for vector of size %u", index, vector->size);
        sc_value_t invalid_value;
        invalid_value.type = -1; // Invalid type
        return invalid_value;
    }

    sc_value_t value;
    value.type = vector->type;

    switch (vector->type) {
        case sc_float16:
            value.value.f16 = ((__bf16*)vector->data)[index];
            break;
        case sc_float32:
            value.value.f32 = ((float*)vector->data)[index];
            break;
        case sc_float64:
            value.value.f64 = ((double*)vector->data)[index];
            break;
        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", vector->type);
            value.type = -1; // Invalid type
            break;
    }

    return value;
}

void sc_set_vector_element(sc_vector* vector, uint64_t index, sc_value_t value) {
    if (index >= vector->size) {
        CCB_ERROR("Index %u out of bounds for vector of size %u", index, vector->size);
        return;
    }

    if (value.type != vector->type) {
        CCB_WARNING("Type mismatch: vector type is %s but value type is %s. Converting value.",
                    sc_TYPES_NAMES[vector->type], sc_TYPES_NAMES[value.type]);
        value = sc_value_as(value, vector->type);
    }

    switch (vector->type) {
        case sc_float16:
            ((__bf16*)vector->data)[index] = value.value.f16;
            break;
        case sc_float32:
            ((float*)vector->data)[index] = value.value.f32;
            break;
        case sc_float64:
            ((double*)vector->data)[index] = value.value.f64;
            break;
        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", vector->type);
            break;
    }
}

sc_vector* sc_get_vector_slice(sc_vector* vector, sc_slice* slice, ccb_arena* arena) {
    if (slice->count != 1) {
        CCB_ERROR("Slice count %u is not supported for vectors (only 1D slices are supported)", slice->count);
        return NULL;
    }

    uint64_t start = slice->slices[0].start;
    uint64_t end = slice->slices[0].end;

    if (start >= vector->size || end > vector->size || start >= end) {
        CCB_ERROR("Invalid slice range [%u, %u) for vector of size %u", start, end, vector->size);
        return NULL;
    }

    uint64_t new_size = end - start;
    sc_vector* sub_vector = sc_create_vector(new_size, vector->type, arena);
    CCB_NOTNULL(sub_vector, "Failed to create sub vector");

    size_t type_size;
    switch (vector->type) {
        case sc_float16:
            type_size = 2;
            break;
        case sc_float32:
            type_size = 4;
            break;
        case sc_float64:
            type_size = 8;
            break;
        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", vector->type);
            return NULL;
    }

    memcpy(sub_vector->data, (unsigned char*)vector->data + start * type_size, new_size * type_size);

    return sub_vector;
}



// tensor functions
sc_value_t sc_get_tensor_element(sc_tensor* tensor, sc_index* index) {
    if (index->count != tensor->dims->dims_count) {
        CCB_ERROR("Index count %u does not match tensor dimensions count %u", index->count, tensor->dims->dims_count);
        sc_value_t invalid_value;
        invalid_value.type = -1; // Invalid type
        return invalid_value;
    }

    uint64_t flat_index = 0;
    uint64_t stride = 1;
    for (int32_t i = tensor->dims->dims_count - 1; i >= 0; i--) {
        if (index->indices[i] >= tensor->dims->dims[i]) {
            CCB_ERROR("Index %u out of bounds for dimension %u of size %u", index->indices[i], i, tensor->dims->dims[i]);
            sc_value_t invalid_value;
            invalid_value.type = -1; // Invalid type
            return invalid_value;
        }
        flat_index += index->indices[i] * stride;
        stride *= tensor->dims->dims[i];
    }

    sc_value_t value;
    value.type = tensor->type;

    switch (tensor->type) {
        case sc_float16:
            value.value.f16 = ((__bf16*)tensor->data)[flat_index];
            break;
        case sc_float32:
            value.value.f32 = ((float*)tensor->data)[flat_index];
            break;
        case sc_float64:
            value.value.f64 = ((double*)tensor->data)[flat_index];
            break;
        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", tensor->type);
            value.type = -1; // Invalid type
            break;
    }

    return value;
}



sc_tensor* sc_get_sub_tensor(sc_tensor* tensor, sc_index* index, ccb_arena* arena) {
    if (index->count >= tensor->dims->dims_count) {

        return NULL;
    }

    for (uint64_t i = 0; i < index->count; i++) {
        if (index->indices[i] >= tensor->dims->dims[i]) {
            CCB_ERROR("Index %u out of bounds for dimension %u of size %u", index->indices[i], i, tensor->dims->dims[i]);
            return NULL;
        }
    }

    uint64_t dims_count = tensor->dims->dims_count - index->count;
    uint64_t* sub_dims = (uint64_t*)ccb_arena_malloc(arena, dims_count * sizeof(uint64_t));
    CCB_NOTNULL(sub_dims, "Failed to allocate memory for sub tensor dimensions");
    
    for (uint64_t i = 0; i < dims_count; i++) {
        sub_dims[i] = tensor->dims->dims[i + index->count];
    }

    sc_dimensions* sub_dims_struct = sc_create_dimensions(dims_count, arena, sub_dims);
    CCB_NOTNULL(sub_dims_struct, "Failed to create sub tensor dimensions struct");

    sc_tensor* sub_tensor = sc_create_tensor(sub_dims_struct, tensor->type, arena);
    CCB_NOTNULL(sub_tensor, "Failed to create sub tensor");
    
    sc_index* source_index = sc_create_empty_index(tensor->dims->dims_count, arena);
    CCB_NOTNULL(source_index, "Failed to create source index");

    for (uint64_t i = 0; i < index->count; i++) {
        source_index->indices[i] = index->indices[i];
    }


    sc_index* target_index = sc_create_empty_index(sub_tensor->dims->dims_count, arena);
    CCB_NOTNULL(target_index, "Failed to create target index");


    for (uint64_t i = 0; i < sub_tensor->size; i++) {
        for (uint64_t j = 0; j < sub_tensor->dims->dims_count; j++) {
            
            uint64_t num = 1;
            for (uint64_t k = 0; k < j; k++) {
                num *= sub_tensor->dims->dims[k];
            }
            
            target_index->indices[j] = (i / (num)) % sub_tensor->dims->dims[j];
            source_index->indices[j+index->count] = target_index->indices[j];
        }
        sc_value_t value = sc_get_tensor_element(tensor, source_index);
        sc_set_tensor_element(sub_tensor, target_index, value);
    }
    return sub_tensor;
}

void sc_set_tensor_element(sc_tensor* tensor, sc_index* index, sc_value_t value) {
    if (index->count != tensor->dims->dims_count) {
        CCB_ERROR("Index count %u does not match tensor dimensions count %u", index->count, tensor->dims->dims_count);
        return;
    }
    
    uint64_t flat_index = 0;
    uint64_t stride = 1;
    
    for (int32_t i = tensor->dims->dims_count - 1; i >= 0; i--) {
        if (index->indices[i] >= tensor->dims->dims[i]) {
            CCB_ERROR("Index %u out of bounds for dimension %u of size %u", index->indices[i], i, tensor->dims->dims[i]);
            return;
        }
        flat_index += index->indices[i] * stride;
        stride *= tensor->dims->dims[i];
    }
    
    if (value.type != tensor->type) {
        CCB_WARNING("Type mismatch: tensor type is %s but value type is %s. Converting value.",
                    sc_TYPES_NAMES[tensor->type], sc_TYPES_NAMES[value.type]);
        value = sc_value_as(value, tensor->type);
    }  
    
    switch (tensor->type) {
        case sc_float16:
            ((__bf16*)tensor->data)[flat_index] = value.value.f16;
            break;
        case sc_float32:
            ((float*)tensor->data)[flat_index] = value.value.f32;
            break;
        case sc_float64:
            ((double*)tensor->data)[flat_index] = value.value.f64;
            break;
        default:
            CCB_ERROR("Unsupported sc_TYPES value %d", tensor->type);
            break;
    }
}


sc_tensor* sc_get_tensor_slice(sc_tensor* tensor, sc_slice* slice, ccb_arena* arena) {
    if (slice->count != tensor->dims->dims_count) {
        CCB_ERROR("Slice count %u does not match tensor dimensions count %u", slice->count, tensor->dims->dims_count);
        return NULL;
    }

    uint64_t* new_dims = (uint64_t*)ccb_arena_malloc(arena, tensor->dims->dims_count * sizeof(uint64_t));
    CCB_NOTNULL(new_dims, "Failed to allocate memory for new tensor dimensions");

    for (uint64_t i = 0; i < slice->count; i++) {
        uint64_t start = slice->slices[i].start;
        uint64_t end = slice->slices[i].end;

        if (start >= tensor->dims->dims[i] || end > tensor->dims->dims[i] || start >= end) {
            CCB_ERROR("Invalid slice range [%u, %u) for dimension %u of size %u", start, end, i, tensor->dims->dims[i]);
            return NULL;
        }

        new_dims[i] = end - start;
    }

    sc_dimensions* new_dims_struct = sc_create_dimensions(tensor->dims->dims_count, arena, new_dims);
    CCB_NOTNULL(new_dims_struct, "Failed to create new tensor dimensions struct");

    sc_tensor* sub_tensor = sc_create_tensor(new_dims_struct, tensor->type, arena);
    CCB_NOTNULL(sub_tensor, "Failed to create sub tensor");

    sc_index* source_index = sc_create_empty_index(tensor->dims->dims_count, arena);
    CCB_NOTNULL(source_index, "Failed to create source index");

    sc_index* target_index = sc_create_empty_index(sub_tensor->dims->dims_count, arena);
    CCB_NOTNULL(target_index, "Failed to create target index");

    for (uint64_t i = 0; i < sub_tensor->size; i++) {
        for (uint64_t j = 0; j < sub_tensor->dims->dims_count; j++) {
            uint64_t num = 1;
            for (uint64_t k = 0; k < j; k++) {
                num *= sub_tensor->dims->dims[k];
            }
            target_index->indices[j] = (i / (num)) % sub_tensor->dims->dims[j];
            source_index->indices[j] = target_index->indices[j] + slice->slices[j].start;
        }
        
        sc_value_t value = sc_get_tensor_element(tensor, source_index);
        sc_set_tensor_element(sub_tensor, target_index, value);
    }

    return sub_tensor;
}


