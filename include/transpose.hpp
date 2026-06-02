void transpose_cpu(const float* mat, float* matT, size_t m, size_t n);
void launch_transpose_naive(const float* mat_dev, float* matT_dev, size_t rows, size_t cols);