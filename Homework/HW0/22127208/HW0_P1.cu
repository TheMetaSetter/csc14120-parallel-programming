#include <stdio.h>

/*
- GPU card’s name
- GPU computation capabilities
- Maximum number of block dimensions
- Maximum number of grid dimensions
- Maximum size of GPU memory
- Amount of constant and share memory
- Warp size
*/
void printDeviceInfo() {
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    printf("**********GPU info**********\n");

    printf("GPU card's name: %s\n", devProp.name);
    printf("GPU computation capability: %d.%d\n", devProp.major, devProp.minor);
    printf("Maximum number of block dimensions:\n");
    printf("Maximum threads along the x-axis: %d\n", devProp.maxThreadsDim[0]);
    printf("Maximum threads along the y-axis: %d\n", devProp.maxThreadsDim[1]);
    printf("Maximum threads along the z-axis: %d\n", devProp.maxThreadsDim[2]);
    printf("Maximum number of grid dimensions:\n");
    printf("Maximum blocks along the x-axis: %d\n", devProp.maxGridSize[0]);
    printf("Maximum blocks along the y-axis: %d\n", devProp.maxGridSize[1]);
    printf("Maximum blocks along the z-axis: %d\n", devProp.maxGridSize[2]);
    printf("Maximum size of GPU memory: %zu bytes\n", devProp.totalGlobalMem);
    printf("Amount of constant memory: %zu bytes\n", devProp.totalConstMem);
    printf("Amount of shared memory: %zu bytes\n", devProp.sharedMemPerBlock);
    printf("Warp size: %d threads\n", devProp.warpSize);

    printf("****************************\n");
}

int main() {
    printDeviceInfo();
    return 0;
}