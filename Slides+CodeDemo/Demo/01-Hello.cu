#include <stdio.h>

// Hàm C có qualifier tên là __global__ được gọi là kernel trong CUDA.
// Khi kernel được gọi (launch), GPU sẽ tạo ra nhiều thread để chạy song song cùng một đoạn mã kernel.
// Mỗi thread chạy cùng một kernel code nhưng xử lý dữ liệu khác nhau trên các CUDA core.
__global__ void helloFromGPU()
{
    printf("Hello from GPU, threadId %d!\n", threadIdx.x);
    printf("Goodbye from GPU, threadId %d!\n", threadIdx.x);
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    cudaGetDeviceProperties(&devProv, 0);
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("****************************\n");
}

int main(int argc, char **argv)
{
    printf("Hello from CPU!\n");

    printDeviceInfo();
    helloFromGPU<<<1, 64>>>(); // 1 group of 64 threads do this function in parallel
    cudaDeviceReset(); // Force to print
    return 0;
}