void reduce_cpu(const float* x, float& out, size_t n);
void reduce_cpu_omp(const float* x, float& out, size_t n);
void reduce_cpu_fp64_ref(const float* x, double& out, size_t n);
void launch_reduce_atomic(const float* x_dev, float* out_dev, size_t n, int block_size = 256);
void launch_reduce_block(const float* x_dev, float* partial_sum_dev, size_t n, int block_size);
void launch_reduce_grid_stride_block(const float* x_dev, float* partial_sum_dev, size_t n,
                                     int block_size, int grid_size);
void launch_reduce_grid_stride_two_pass(const float* x_dev, float* partial_sum_dev, float* out_dev,
                                        size_t n, int block_size, int grid_size);
void launch_reduce_multi_pass(const float* x_dev, float* partial_sum_dev, float* scratch_dev,
                              float* out_dev, size_t n, int block_size, int grid_size);
void launch_reduce_warp_shuffle(const float* x_dev, float* partial_sum_dev, size_t n,
                                int block_size, int grid_size);