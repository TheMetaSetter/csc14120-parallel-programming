#include <stdio.h>
#define N 16777216
#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);                                                                 
		cudaEventSynchronize(start);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

void addVecOnHost(float* in_vec_1, float* in_vec_2, float* out_vec, int vec_size)
{
    for (int i = 0; i < vec_size; i++)
        out_vec[i] = in_vec_1[i] + in_vec_2[i];
}

__global__ void addVecOnDeviceVer1(float* in_vec_1, float* in_vec_2, float* out_vec, int vec_size) {
    // Get index of the thread running this kernel on these specific data
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread handle 2 indices (i and i2) of the output vector
    if (i < vec_size - 2*blockDim.x + 1) {
        out_vec[i] = in_vec_1[i] + in_vec_2[i];
        int i2 = i + 2*blockDim.x - 1;
        out_vec[i2] = in_vec_1[i2] + in_vec_2[i2];
    }
}

__global__ void addVecOnDeviceVer2(const float* in_vec_1, const float* in_vec_2,
                                   float* out_vec, int vec_size) {
    int i  = blockIdx.x * blockDim.x + threadIdx.x;
    int i1 = 2*i;
    int i2 = i1 + 1;

    if (i1 < vec_size) out_vec[i1] = in_vec_1[i1] + in_vec_2[i1];
    if (i2 < vec_size) out_vec[i2] = in_vec_1[i2] + in_vec_2[i2];
}

void addVec(float* in_vec_1, float* in_vec_2, float* out_vec, int vec_size,
            bool useDevice = false, bool ver1 = true) {
    GpuTimer timer;

    if (useDevice == false) {
        timer.Start();
        addVecOnHost(in_vec_1, in_vec_2, out_vec, vec_size);
        timer.Stop();
    } else {
        // Show basic information of the current device
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, 0);
        printf("GPU name: %s\n", devProp.name);
        printf("GPU compute capability: %d.%d\n", devProp.major, devProp.minor);

        // CPU (host) allocates memory on device
        float *d_in1, *d_in2, *d_out;                // Declare pointers to vectors on the device's memory
        size_t numBytes = vec_size * sizeof(float);  // Size of each vector in bytes
        CHECK(cudaMalloc(&d_in1, numBytes));
        CHECK(cudaMalloc(&d_in2, numBytes));
        CHECK(cudaMalloc(&d_out, numBytes));

        // Host copies data from host memory to device memory
        CHECK(cudaMemcpy(d_in1, in_vec_1, numBytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_in2, in_vec_2, numBytes, cudaMemcpyHostToDevice));

        // Host invokes kernel function to add vectors on device
        dim3 blockSize(256);
        dim3 gridSize((vec_size - 1) / blockSize.x + 1);  // TODO: Tại sao cấp phát nhiêu đây block cho 1 grid?

        // Launch kernel and measure the processing time of the GPU
        if (ver1) {
            timer.Start();
            addVecOnDeviceVer1<<<gridSize, blockSize>>>(d_in1, d_in2, d_out, vec_size);
            cudaDeviceSynchronize();
            timer.Stop();
        } else {
            timer.Start();
            addVecOnDeviceVer2<<<gridSize, blockSize>>>(d_in1, d_in2, d_out, vec_size);
            cudaDeviceSynchronize();
            timer.Stop();
        }

        // Host copies result from device memory
        CHECK(cudaMemcpy(out_vec, d_out, numBytes, cudaMemcpyDeviceToHost));

        // Free device memories
        CHECK(cudaFree(d_in1));
        CHECK(cudaFree(d_in2));
        CHECK(cudaFree(d_out));
    }

    // Print out processing time
    float time = timer.Elapsed();
    printf("Vector size: %d, processing time (%s): %f ms, %s\n",
        vec_size,
        useDevice ? "use device" : "use host",
        time,
        ver1 ? "add vec ver1." : "add vec ver2.");
}

int main(int argc, char** argv) {
    // Array of vector sizes for benchmarking
    int vec_sizes[] = {64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216};
    int num_sizes = sizeof(vec_sizes) / sizeof(vec_sizes[0]);

    // For each version (ver1 and ver2)
    for (int ver = 0; ver < 2; ++ver) {
        bool ver1 = (ver == 0);
        printf("==== Running %s ====\n", ver1 ? "addVecVer1" : "addVecVer2");
        for (int s = 0; s < num_sizes; ++s) {
            int currN = vec_sizes[s];
            size_t numBytes = currN * sizeof(float);

            // Allocate memory for input and output vectors
            float* in_vec_1 = (float*)(malloc(numBytes));
            float* in_vec_2 = (float*)(malloc(numBytes));
            float* out_vec = (float*)(malloc(numBytes));
            float* correct_out_vec = (float*)(malloc(numBytes));

            // Initialize input vectors
            for (int i = 0; i < currN; ++i) {
                in_vec_1[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                in_vec_2[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            }

            // Host computation (baseline)
            addVec(in_vec_1, in_vec_2, correct_out_vec, currN);

            // Device computation
            addVec(in_vec_1, in_vec_2, out_vec, currN, true, ver1);

            // Check correctness
            bool correct = true;
            for (int i = 0; i < currN; ++i) {
                if (out_vec[i] != correct_out_vec[i]) {
                    correct = false;
                    break;
                }
            }
            printf("Vector size: %d, Version: %s, %s\n", currN, ver1 ? "ver1" : "ver2", correct ? "CORRECT :)" : "INCORRECT :(");
            if (!correct) {
                // Free memory before exit
                free(in_vec_1);
                free(in_vec_2);
                free(out_vec);
                free(correct_out_vec);
                return 1;
            }

            // Free memory for this experiment
            free(in_vec_1);
            free(in_vec_2);
            free(out_vec);
            free(correct_out_vec);
        }
    }
    return 0;
}