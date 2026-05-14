#pragma once
void vector_add_cpu(const float* a, const float* b, float* c, int num_elem);
void launch_vector_add(const float* a_dev, const float* b_dev, float* c_dev, int num_elem);