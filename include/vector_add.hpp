#pragma once
void vector_add_cpu(const float* a, const float* b, float* c, size_t num_elem);
void launch_vector_add(const float* a_dev, const float* b_dev, float* c_dev, size_t num_elem);