void reduce_cpu(const float* input, float& output, size_t n);
void launch_reduce_atomic(const float* x_dev, float* out_dev, size_t n, int block_size = 256);
void launch_reduce_block(const float* x_dev, float* partial_sum_dev, size_t n, int block_size);
void launch_reduce_grid_stride_block(const float* x_dev, float* partial_sum_dev, size_t n,
                                     int block_size, int grid_size);