#include "data.h"
#include "const.h"
#include "../ccbase/logs/log.h"

#include <stdlib.h>
#include <string.h>


char** CL_TYPES_NAMES = (char*[]) {
    "float16",
    "float32",
    "float64"
};



// Create data structures
cl_dimensions* cl_create_empty_dimensions(uint32_t dims_count, ccb_arena* arena) {
    cl_dimensions* dimensions = (cl_dimensions*)ccb_arena_malloc(arena, sizeof(cl_dimensions));
    CCB_NOTNULL(dimensions, "Failed to allocate memory for dimensions struct");

    dimensions->dims_count = dims_count;
    dimensions->dims = (uint32_t*)ccb_arena_malloc(arena, dims_count * sizeof(uint32_t));
    CCB_NOTNULL(dimensions->dims, "Failed to allocate memory for dimensions array");

    return dimensions;
}


cl_slice_t* cl_create_empty_slice(uint32_t count, ccb_arena* arena) {
    cl_slice_t* slice = (cl_slice_t*)ccb_arena_malloc(arena, sizeof(cl_slice_t));
    CCB_NOTNULL(slice, "Failed to allocate memory for slice struct");

    slice->count = count;
    slice->slices = (cl_slice_el_t*)ccb_arena_malloc(arena, count * sizeof(cl_slice_el_t));
    CCB_NOTNULL(slice->slices, "Failed to allocate memory for slice array");

    return slice;
}


cl_index* cl_create_empty_index(uint32_t count, ccb_arena* arena) {
    cl_index* index = (cl_index*)ccb_arena_malloc(arena, sizeof(cl_index));
    CCB_NOTNULL(index, "Failed to allocate memory for index struct");

    index->count = count;
    index->indices = (uint32_t*)ccb_arena_malloc(arena, count * sizeof(uint32_t));
    CCB_NOTNULL(index->indices, "Failed to allocate memory for index array");

    return index;
}



cl_vector* cl_create_vector(uint32_t size, CL_TYPES type, ccb_arena* arena) {
    cl_vector* vector = (cl_vector*)ccb_arena_malloc(arena, sizeof(cl_vector));
    CCB_NOTNULL(vector, "Failed to allocate memory for vector struct");

    vector->size = size;
    vector->type = type;

    size_t type_size;
    switch (type) {
        case CL_float16:
            type_size = 2;
            break;
        case CL_float32:
            type_size = 4;
            break;
        case CL_float64:
            type_size = 8;
            break;
        default:
            CCB_ERROR("Unsupported CL_TYPES value %d", type);
            return NULL;
    }

    vector->data = ccb_arena_malloc(arena, size * type_size);
    CCB_NOTNULL(vector->data, "Failed to allocate memory for vector data");

    return vector;    
}



cl_dimensions* cl_create_dimensions(uint32_t dims_count, ccb_arena* arena, uint32_t* dims) {
    cl_dimensions* dimensions = (cl_dimensions*)ccb_arena_malloc(arena, sizeof(cl_dimensions));
    CCB_NOTNULL(dimensions, "Failed to allocate memory for dimensions struct");

    dimensions->dims_count = dims_count;
    dimensions->dims = (uint32_t*)ccb_arena_malloc(arena, dims_count * sizeof(uint32_t));
    CCB_NOTNULL(dimensions->dims, "Failed to allocate memory for dimensions array");

    for (uint32_t i = 0; i < dims_count; i++) {
        dimensions->dims[i] = dims[i];
    }

    return dimensions;
}


cl_index* cl_create_index(uint32_t count, ccb_arena* arena, uint32_t* indices) {
    cl_index* index = (cl_index*)ccb_arena_malloc(arena, sizeof(cl_index));
    CCB_NOTNULL(index, "Failed to allocate memory for index struct");

    index->count = count;
    index->indices = (uint32_t*)ccb_arena_malloc(arena, count * sizeof(uint32_t));
    CCB_NOTNULL(index->indices, "Failed to allocate memory for index array");

    for (uint32_t i = 0; i < count; i++) {
        index->indices[i] = indices[i];
    }

    return index;
}

cl_slice_t* cl_create_slice(uint32_t count, ccb_arena* arena, uint32_t* starts, uint32_t* ends) {
    CCB_NOTNULL(arena, "Invalid arena pointer");
    CCB_NOTNULL(starts, "Invalid starts pointer");
    CCB_NOTNULL(ends, "Invalid ends pointer");

    cl_slice_t* slice = (cl_slice_t*)ccb_arena_malloc(arena, sizeof(cl_slice_t));
    CCB_NOTNULL(slice, "Failed to allocate memory for slice struct");

    slice->count = count;
    slice->slices = (cl_slice_el_t*)ccb_arena_malloc(arena, count * sizeof(cl_slice_el_t));
    CCB_NOTNULL(slice->slices, "Failed to allocate memory for slice array");

    for (uint32_t i = 0; i < count; i++) {
        slice->slices[i].start = starts[i];
        slice->slices[i].end = ends[i];
    }

    return slice;
}


cl_tensor* cl_create_tensor(cl_dimensions* dims, CL_TYPES type, ccb_arena* arena) {
    cl_tensor* tensor = (cl_tensor*)ccb_arena_malloc(arena, sizeof(cl_tensor));
    CCB_NOTNULL(tensor, "Failed to allocate memory for tensor struct");

    tensor->size = 1;
    for (uint32_t i = 0; i < dims->dims_count; i++) {
        tensor->size *= dims->dims[i];
    }

    tensor->type = type;

    size_t type_size;
    switch (type) {
        case CL_float16:
            type_size = 2;
            break;
        case CL_float32:
            type_size = 4;
            break;
        case CL_float64:
            type_size = 8;
            break;
        default:
            CCB_ERROR("Unsupported CL_TYPES value %d", type);
            return NULL;
    }
    tensor->dims = dims;

    tensor->data = ccb_arena_malloc(arena, tensor->size * type_size);
    CCB_NOTNULL(tensor->data, "Failed to allocate memory for tensor data");

    return tensor;    
}

// clone functions
cl_vector* cl_clone_vector(cl_vector* vector, ccb_arena* arena) {
    cl_vector* clone = (cl_vector*)ccb_arena_malloc(arena, sizeof(cl_vector));
    CCB_NOTNULL(clone, "Failed to allocate memory for clone struct");

    clone->size = vector->size;
    clone->type = vector->type;

    size_t type_size;
    switch (vector->type) {
        case CL_float16:
            type_size = 2;
            break;
        case CL_float32:
            type_size = 4;
            break;
        case CL_float64:
            type_size = 8;
            break;
        default:
            CCB_ERROR("Unsupported CL_TYPES value %d", vector->type);
            return NULL;
    }

    clone->data = ccb_arena_malloc(arena, vector->size * type_size);
    CCB_NOTNULL(clone->data, "Failed to allocate memory for clone data");

    memcpy(clone->data, vector->data, vector->size * type_size);

    return clone;
}


cl_tensor* cl_clone_tensor(cl_tensor* tensor, ccb_arena* arena) {
    cl_tensor* clone = (cl_tensor*)ccb_arena_malloc(arena, sizeof(cl_tensor));
    CCB_NOTNULL(clone, "Failed to allocate memory for clone struct");

    clone->size = tensor->size;
    clone->type = tensor->type;
    clone->dims = cl_clone_dimensions(tensor->dims, arena);
    CCB_NOTNULL(clone->dims, "Failed to clone dimensions");

    size_t type_size;
    switch (tensor->type) {
        case CL_float16:
            type_size = 2;
            break;
        case CL_float32:
            type_size = 4;
            break;
        case CL_float64:
            type_size = 8;
            break;
        default:
            CCB_ERROR("Unsupported CL_TYPES value %d", tensor->type);
            return NULL;
    }

    clone->data = ccb_arena_malloc(arena, tensor->size * type_size);
    CCB_NOTNULL(clone->data, "Failed to allocate memory for clone data");

    memcpy(clone->data, tensor->data, tensor->size * type_size);

    return clone;
}

cl_index* cl_clone_index(cl_index* index, ccb_arena* arena) {
    cl_index* clone = (cl_index*)ccb_arena_malloc(arena, sizeof(cl_index));
    CCB_NOTNULL(clone, "Failed to allocate memory for clone struct");

    clone->count = index->count;

    clone->indices = (uint32_t*)ccb_arena_malloc(arena, index->count * sizeof(uint32_t));
    CCB_NOTNULL(clone->indices, "Failed to allocate memory for clone indices");

    memcpy(clone->indices, index->indices, index->count * sizeof(uint32_t));

    return clone;
}

cl_slice_t* cl_clone_slice(cl_slice_t* slice, ccb_arena* arena) {
    cl_slice_t* clone = (cl_slice_t*)ccb_arena_malloc(arena, sizeof(cl_slice_t));
    CCB_NOTNULL(clone, "Failed to allocate memory for clone struct");

    clone->count = slice->count;

    clone->slices = (cl_slice_el_t*)ccb_arena_malloc(arena, slice->count * sizeof(cl_slice_el_t));
    CCB_NOTNULL(clone->slices, "Failed to allocate memory for clone slices");

    memcpy(clone->slices, slice->slices, slice->count * sizeof(cl_slice_el_t));

    return clone;
}


cl_dimensions* cl_clone_dimensions(cl_dimensions* dimensions, ccb_arena* arena) {
    cl_dimensions* clone = (cl_dimensions*)ccb_arena_malloc(arena, sizeof(cl_dimensions));
    CCB_NOTNULL(clone, "Failed to allocate memory for clone struct");

    clone->dims_count = dimensions->dims_count;

    clone->dims = (uint32_t*)ccb_arena_malloc(arena, dimensions->dims_count * sizeof(uint32_t));
    CCB_NOTNULL(clone->dims, "Failed to allocate memory for clone dims");

    memcpy(clone->dims, dimensions->dims, dimensions->dims_count * sizeof(uint32_t));

    return clone;
}

// Load data
void cl_data_to_vector(cl_vector* vector, void* data, uint32_t size) {
    size_t type_size;
    switch (vector->type) {
        case CL_float16:
            type_size = 2;
            break;
        case CL_float32:
            type_size = 4;
            break;
        case CL_float64:
            type_size = 8;
            break;
        default:
            CCB_ERROR("Unsupported CL_TYPES value %d", vector->type);
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


void cl_data_to_tensor(cl_tensor* tensor, void* data, uint32_t size) {
    size_t type_size;
    switch (tensor->type) {
        case CL_float16:
            type_size = 2;
            break;
        case CL_float32:
            type_size = 4;
            break;
        case CL_float64:
            type_size = 8;
            break;
        default:
            CCB_ERROR("Unsupported CL_TYPES value %d", tensor->type);
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
void cl_print_vector(cl_vector* vector) {
    size_t type_size;
    switch (vector->type) {
        case CL_float16:
            type_size = 2;
            break;
        case CL_float32:
            type_size = 4;
            break;
        case CL_float64:
            type_size = 8;
            break;
        default:
            CCB_ERROR("Unsupported CL_TYPES value %d", vector->type);
            return;
    }

    printf("Vector (size: %u, type: %s):\n", vector->size, CL_TYPES_NAMES[vector->type]);
    for (uint32_t i = 0; i < vector->size; i++) {
        if (vector->type == CL_float32) {
            float* data = (float*)vector->data;
            printf("  [%u]: %f\n", i, data[i]);
        } else if (vector->type == CL_float64) {
            double* data = (double*)vector->data;
            printf("  [%u]: %lf\n", i, data[i]);
        } else if (vector->type == CL_float16) {
            __bf16* data = (__bf16*)vector->data;
            printf("  [%u]: %f\n", i, (float)data[i]);
        } else {
            printf("  [%u]: <unsupported type for printing>\n", i);
        }
    }
}


void cl_print_tensor(cl_tensor* tensor, ccb_arena* tmp_arena) {
    size_t type_size;
    switch (tensor->type) {
        case CL_float16:
            type_size = 2;
            break;
        case CL_float32:
            type_size = 4;
            break;
        case CL_float64:
            type_size = 8;
            break;
        default:
            CCB_ERROR("Unsupported CL_TYPES value %d", tensor->type);
            return;
    }

    cl_index index;
    index.count = tensor->dims->dims_count;
    index.indices = (uint32_t*)ccb_arena_malloc(tmp_arena, tensor->dims->dims_count * sizeof(uint32_t));
    CCB_NOTNULL(index.indices, "Failed to allocate memory for index array");
    memset(index.indices, 0, tensor->size * sizeof(uint32_t));

    printf("tensor (size: %u, type: %s, dimensions: [", tensor->size, CL_TYPES_NAMES[tensor->type]);
    for (uint32_t i = 0; i < tensor->dims->dims_count; i++) {
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

    for (uint32_t i = 0; i < tensor->size; i++) {
        printf("  [");
        for (uint32_t j = 0; j < tensor->dims->dims_count; j++) {
            printf("%u", index.indices[j]);
            if (j < tensor->dims->dims_count - 1) {
                printf(", ");
            }
        }
        printf("]: ");

        if (tensor->type == CL_float32) {
            float* data = (float*)tensor->data;
            printf("%f\n", data[i]);
        } else if (tensor->type == CL_float64) {
            double* data = (double*)tensor->data;
            printf("%lf\n", data[i]);
        } else if (tensor->type == CL_float16) {
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



// Values functions
cl_value_t to_cl_value(double value, CL_TYPES type) {
    cl_value_t scalar;
    scalar.type = type;
    switch (type) {
        case CL_float16:
            scalar.value.f16 = (__bf16)value;
            break;
        case CL_float32:
            scalar.value.f32 = (float)value;
            break;
        case CL_float64:
            scalar.value.f64 = value;
            break;
        default:
            CCB_ERROR("Unsupported CL_TYPES value %d", type);
            scalar.type = -1; // Invalid type
            break;
    }
    return scalar;
}


__bf16 cl_value_to_f16(cl_value_t value) {
    switch (value.type) {
        case CL_float16:
            return value.value.f16;
        case CL_float32:
            return (__bf16)value.value.f32;
        case CL_float64:
            return (__bf16)value.value.f64;
        default:
            CCB_ERROR("Unsupported CL_TYPES value %d", value.type);
            return (__bf16)0.0; // Default return value
    }
}


float cl_value_to_f32(cl_value_t value) {
    switch (value.type) {
        case CL_float16:
            return (float)value.value.f16;
        case CL_float32:
            return value.value.f32;
        case CL_float64:
            return (float)value.value.f64;
        default:
            CCB_ERROR("Unsupported CL_TYPES value %d", value.type);
            return 0.0f; // Default return value
    }
}


double cl_value_to_f64(cl_value_t value) {
    switch (value.type) {
        case CL_float16:
            return (double)value.value.f16;
        case CL_float32:
            return (double)value.value.f32;
        case CL_float64:
            return value.value.f64;
        default:
            CCB_ERROR("Unsupported CL_TYPES value %d", value.type);
            return 0.0; // Default return value
    }
}


cl_value_t cl_value_as(cl_value_t a, CL_TYPES target_type) {
    double value = cl_value_to_f64(a); // Convert to double first for precision
    return to_cl_value(value, target_type);
}


// geters and seters
cl_value_t cl_get_vector_element(cl_vector* vector, uint32_t index) {
    if (index >= vector->size) {
        CCB_ERROR("Index %u out of bounds for vector of size %u", index, vector->size);
        cl_value_t invalid_value;
        invalid_value.type = -1; // Invalid type
        return invalid_value;
    }

    cl_value_t value;
    value.type = vector->type;

    switch (vector->type) {
        case CL_float16:
            value.value.f16 = ((__bf16*)vector->data)[index];
            break;
        case CL_float32:
            value.value.f32 = ((float*)vector->data)[index];
            break;
        case CL_float64:
            value.value.f64 = ((double*)vector->data)[index];
            break;
        default:
            CCB_ERROR("Unsupported CL_TYPES value %d", vector->type);
            value.type = -1; // Invalid type
            break;
    }

    return value;
}

void cl_set_vector_element(cl_vector* vector, uint32_t index, cl_value_t value) {
    if (index >= vector->size) {
        CCB_ERROR("Index %u out of bounds for vector of size %u", index, vector->size);
        return;
    }

    if (value.type != vector->type) {
        CCB_WARNING("Type mismatch: vector type is %s but value type is %s. Converting value.",
                    CL_TYPES_NAMES[vector->type], CL_TYPES_NAMES[value.type]);
        value = cl_value_as(value, vector->type);
    }

    switch (vector->type) {
        case CL_float16:
            ((__bf16*)vector->data)[index] = value.value.f16;
            break;
        case CL_float32:
            ((float*)vector->data)[index] = value.value.f32;
            break;
        case CL_float64:
            ((double*)vector->data)[index] = value.value.f64;
            break;
        default:
            CCB_ERROR("Unsupported CL_TYPES value %d", vector->type);
            break;
    }
}

cl_value_t cl_get_tensor_element(cl_tensor* tensor, cl_index* index) {
    if (index->count != tensor->dims->dims_count) {
        CCB_ERROR("Index count %u does not match tensor dimensions count %u", index->count, tensor->dims->dims_count);
        cl_value_t invalid_value;
        invalid_value.type = -1; // Invalid type
        return invalid_value;
    }

    uint32_t flat_index = 0;
    uint32_t stride = 1;
    for (int32_t i = tensor->dims->dims_count - 1; i >= 0; i--) {
        if (index->indices[i] >= tensor->dims->dims[i]) {
            CCB_ERROR("Index %u out of bounds for dimension %u of size %u", index->indices[i], i, tensor->dims->dims[i]);
            cl_value_t invalid_value;
            invalid_value.type = -1; // Invalid type
            return invalid_value;
        }
        flat_index += index->indices[i] * stride;
        stride *= tensor->dims->dims[i];
    }

    cl_value_t value;
    value.type = tensor->type;

    switch (tensor->type) {
        case CL_float16:
            value.value.f16 = ((__bf16*)tensor->data)[flat_index];
            break;
        case CL_float32:
            value.value.f32 = ((float*)tensor->data)[flat_index];
            break;
        case CL_float64:
            value.value.f64 = ((double*)tensor->data)[flat_index];
            break;
        default:
            CCB_ERROR("Unsupported CL_TYPES value %d", tensor->type);
            value.type = -1; // Invalid type
            break;
    }

    return value;
}



cl_tensor* cl_get_sub_tensor(cl_tensor* tensor, cl_index* index, ccb_arena* arena) {
    if (index->count > tensor->dims->dims_count) {
        return NULL;
    }

    for (uint32_t i = 0; i < index->count; i++) {
        if (index->indices[i] >= tensor->dims->dims[i]) {
            CCB_ERROR("Index %u out of bounds for dimension %u of size %u", index->indices[i], i, tensor->dims->dims[i]);
            return NULL;
        }
    }

    uint32_t dims_count = tensor->dims->dims_count - index->count;
    uint32_t* sub_dims = (uint32_t*)ccb_arena_malloc(arena, dims_count * sizeof(uint32_t));
    CCB_NOTNULL(sub_dims, "Failed to allocate memory for sub tensor dimensions");
    
    for (uint32_t i = 0; i < dims_count; i++) {
        sub_dims[i] = tensor->dims->dims[i + index->count];
    }

    cl_dimensions* sub_dims_struct = cl_create_dimensions(dims_count, arena, sub_dims);
    CCB_NOTNULL(sub_dims_struct, "Failed to create sub tensor dimensions struct");

    cl_tensor* sub_tensor = cl_create_tensor(sub_dims_struct, tensor->type, arena);
    CCB_NOTNULL(sub_tensor, "Failed to create sub tensor");
    
    cl_index* source_index = cl_create_empty_index(tensor->dims->dims_count, arena);
    CCB_NOTNULL(source_index, "Failed to create source index");

    for (uint32_t i = 0; i < index->count; i++) {
        source_index->indices[i] = index->indices[i];
    }


    cl_index* target_index = cl_create_empty_index(sub_tensor->dims->dims_count, arena);
    CCB_NOTNULL(target_index, "Failed to create target index");


    for (uint32_t i = 0; i < sub_tensor->size; i++) {
        for (uint32_t j = 0; j < sub_tensor->dims->dims_count; j++) {
            target_index->indices[j] = (i / (j == 0 ? 1 : sub_tensor->dims->dims[j - 1])) % sub_tensor->dims->dims[j];
            source_index->indices[j] += target_index->indices[j];
        }
        cl_value_t value = cl_get_tensor_element(tensor, source_index);
        cl_set_tensor_element(sub_tensor, target_index, value);
    }
    return sub_tensor;
}

void cl_set_tensor_element(cl_tensor* tensor, cl_index* index, cl_value_t value) {
    if (index->count != tensor->dims->dims_count) {
        CCB_ERROR("Index count %u does not match tensor dimensions count %u", index->count, tensor->dims->dims_count);
        return;
    }
    
    uint32_t flat_index = 0;
    uint32_t stride = 1;
    
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
                    CL_TYPES_NAMES[tensor->type], CL_TYPES_NAMES[value.type]);
        value = cl_value_as(value, tensor->type);
    }  
    
    switch (tensor->type) {
        case CL_float16:
            ((__bf16*)tensor->data)[flat_index] = value.value.f16;
            break;
        case CL_float32:
            ((float*)tensor->data)[flat_index] = value.value.f32;
            break;
        case CL_float64:
            ((double*)tensor->data)[flat_index] = value.value.f64;
            break;
        default:
            CCB_ERROR("Unsupported CL_TYPES value %d", tensor->type);
            break;
    }
}