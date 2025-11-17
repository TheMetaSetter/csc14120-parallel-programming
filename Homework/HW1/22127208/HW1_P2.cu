#include <stdio.h>
#include <stdint.h>

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

void readPnm(char * fileName, 
		int &width, int &height, uchar3 * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	
	if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);
	
	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	for (int i = 0; i < width * height; i++)
		fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

	fclose(f);
}

void writePnm(uchar3 * pixels, int width, int height, 
		char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	fprintf(f, "P3\n%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height; i++)
		fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);
	
	fclose(f);
}

__global__ void blurImgKernel(const uchar3* __restrict__ inPixels, int width, int height, 
                              const float* __restrict__ filter, int filterWidth, 
                              uchar3* __restrict__ outPixels)
{
	// TODO
	// Calculate the 1D index of the output data of the current thread
	int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (globalThreadIdx >= width * height) {
		return; // Do nothing with this thread since the index is outside the image boundaries
	}

	// Calculate the 2D coordinates of the center pixel
	int centerRow = globalThreadIdx / width;
	int centerCol = globalThreadIdx % width;

	// Use float accumulators for precision
	float redSum = 0.0f;
	float greenSum = 0.0f;
	float blueSum = 0.0f;

	int filterRadius = filterWidth / 2;
	// Perform convolution by iterating over the filter
	// Calculate source pixel coordinates and clamp them on the fly
	for (int i = 0; i < filterWidth; ++i) {
		for (int j = 0 ; j < filterWidth; ++j) {
			// Calculate the coordinates of the source pixel in the image
			int sourceRow = centerRow + i - filterRadius;
			int sourceCol = centerCol + j - filterRadius;

			// Clamp coordinates to handle edges efficiently
			if (sourceRow < 0) sourceRow = 0;
			if (sourceRow >= height) sourceRow = height - 1;
			if (sourceCol < 0) sourceCol = 0;
			if (sourceCol >= width) sourceCol = width - 1;

			// Calculate the 1D index for the source pixel
			int sourceIdx = sourceRow * width + sourceCol;

			// Access the 1D filter array
			float filterWeight = filter[i * filterWidth + j];

			// Apply the filter weight to all color channels
			redSum += inPixels[sourceIdx].x * filterWeight;
			greenSum += inPixels[sourceIdx].y * filterWeight;
			blueSum += inPixels[sourceIdx].z * filterWeight;
		}
	}

	// Write the final result, clamping it to the valid uchar range [0, 255]
	outPixels[globalThreadIdx].x = (uint8_t)__float2uint_rn(fminf(fmaxf(redSum, 0.0f), 255.0f));
    outPixels[globalThreadIdx].y = (uint8_t)__float2uint_rn(fminf(fmaxf(greenSum, 0.0f), 255.0f));
    outPixels[globalThreadIdx].z = (uint8_t)__float2uint_rn(fminf(fmaxf(blueSum, 0.0f), 255.0f));
}

void blurImgOnHost(const uchar3* inPixels, int width, int height, 
                   const float* filter, int filterWidth, 
                   uchar3* outPixels)
{
    int totalPixels = width * height;
    int filterRadius = filterWidth / 2;

    // Loop over every pixel in the output image
    for (int pixelIdx = 0; pixelIdx < totalPixels; ++pixelIdx) {
        // Calculate the 2D coordinates of the center pixel
        int centerRow = pixelIdx / width;
        int centerCol = pixelIdx % width;

        // Use float accumulators for precision during summation
        float redSum = 0.0f;
        float greenSum = 0.0f;
        float blueSum = 0.0f;

        // Perform the convolution by iterating over the filter
        for (int i = 0; i < filterWidth; ++i) {
            for (int j = 0; j < filterWidth; ++j) {
                // Calculate the coordinates of the source pixel in the input image
                int sourceRow = centerRow + i - filterRadius;
                int sourceCol = centerCol + j - filterRadius;

                // Clamp coordinates to handle edges (replicate border pixels)
                if (sourceRow < 0) sourceRow = 0;
                if (sourceRow >= height) sourceRow = height - 1;
                if (sourceCol < 0) sourceCol = 0;
                if (sourceCol >= width) sourceCol = width - 1;

                // Calculate the 1D index for the source pixel
                int srcIdx = sourceRow * width + sourceCol;

                // Get the filter weight from the 1D filter array
                float filterWeight = filter[i * filterWidth + j];

                // Accumulate the weighted sum for each color channel
                redSum += inPixels[srcIdx].x * filterWeight;
                greenSum += inPixels[srcIdx].y * filterWeight;
                blueSum += inPixels[srcIdx].z * filterWeight;
            }
        }
        
        // Write the final result to the output image, clamping to the valid [0, 255] range
        outPixels[pixelIdx].x = (uint8_t)lrintf(fminf(fmaxf(redSum, 0.0f), 255.0f));
        outPixels[pixelIdx].y = (uint8_t)lrintf(fminf(fmaxf(greenSum, 0.0f), 255.0f));
        outPixels[pixelIdx].z = (uint8_t)lrintf(fminf(fmaxf(blueSum, 0.0f), 255.0f));
    }
}

void blurImg(uchar3 * inPixels, int width, int height, float * filter, int filterWidth, 
		uchar3 * outPixels,
		bool useDevice=false, dim3 blockSize=dim3(1, 1))
{
	GpuTimer timer;
	timer.Start();
	if (useDevice == false)
	{
		// TODO
		// Call the host function
		blurImgOnHost(inPixels, width, height, filter, filterWidth, outPixels);
	}
	else // Use device
	{
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, 0);
		printf("GPU name: %s\n", devProp.name);
		printf("GPU compute capability: %d.%d\n", devProp.major, devProp.minor);

		// TODO
		// Allocate device memories
		uchar3* d_inPixels;
		uchar3* d_outPixels;
		float* d_filter;
		size_t imgSize = width * height * sizeof(uchar3);
		size_t filterSize = filterWidth * filterWidth * sizeof(float);
		CHECK(cudaMalloc(&d_inPixels, imgSize));
		CHECK(cudaMalloc(&d_outPixels, imgSize));
		CHECK(cudaMalloc(&d_filter, filterSize));

		// Copy data from host to device
		CHECK(cudaMemcpy(d_inPixels, inPixels, imgSize, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_filter, filter, filterSize, cudaMemcpyHostToDevice));

		// Launch kernel
		int totalPixels = width * height;
		int numBlocks = (totalPixels + blockSize.x - 1) / blockSize.x;
		blurImgKernel<<<numBlocks, blockSize>>>(d_inPixels, width, height, d_filter, filterWidth, d_outPixels);
		CHECK(cudaDeviceSynchronize());
		
		// Copy result from device to host
		CHECK(cudaMemcpy(outPixels, d_outPixels, imgSize, cudaMemcpyDeviceToHost));
		// Free device memories
		CHECK(cudaFree(d_inPixels));
		CHECK(cudaFree(d_outPixels));
		CHECK(cudaFree(d_filter));
	}
	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time (%s): %f ms\n", 
    		useDevice == true? "use device" : "use host", time);
}

float computeError(uchar3 * a1, uchar3 * a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)
	{
		err += abs((int)a1[i].x - (int)a2[i].x);
		err += abs((int)a1[i].y - (int)a2[i].y);
		err += abs((int)a1[i].z - (int)a2[i].z);
	}
	err /= (n * 3);
	return err;
}

char * concatStr(const char * s1, const char * s2)
{
    char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

int main(int argc, char ** argv)
{
	if (argc != 4 && argc != 6)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}

	// Read input image file
	int width, height;
	uchar3 * inPixels;
	readPnm(argv[1], width, height, inPixels);
	printf("Image size (width x height): %i x %i\n\n", width, height);

	// Debug
	printf("argv[1]= %s\n", argv[1]);
	printf("argv[2]= %s\n", argv[2]);
	printf("argv[3]= %s\n", argv[3]);

	// Read correct output image file
	int correctWidth, correctHeight;
	uchar3 * correctOutPixels;
	readPnm(argv[3], correctWidth, correctHeight, correctOutPixels);
	if (correctWidth != width || correctHeight != height)
	{
		printf("The shape of the correct output image is invalid\n");
		return EXIT_FAILURE;
	}

	// Set up a simple filter with blurring effect 
	int filterWidth = 9;
	float * filter = (float *)malloc(filterWidth * filterWidth * sizeof(float));
	for (int filterR = 0; filterR < filterWidth; filterR++)
	{
		for (int filterC = 0; filterC < filterWidth; filterC++)
		{
			filter[filterR * filterWidth + filterC] = 1. / (filterWidth * filterWidth);
		}
	}

	// Blur input image using host
	uchar3 * hostOutPixels = (uchar3 *)malloc(width * height * sizeof(uchar3)); 
	blurImg(inPixels, width, height, filter, filterWidth, hostOutPixels);
	
	// Compute mean absolute error between host result and correct result
	float hostErr = computeError(hostOutPixels, correctOutPixels, width * height);
	printf("Error: %f\n\n", hostErr);

	// Blur input image using device
	uchar3 * deviceOutPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	dim3 blockSize(32, 32); // Default

	if (argc == 6)
	{
		blockSize.x = atoi(argv[4]);
		blockSize.y = atoi(argv[5]);
	}
	blurImg(inPixels, width, height, filter, filterWidth, deviceOutPixels, true, blockSize);

	// Compute mean absolute error between device result and correct result
	float deviceErr = computeError(deviceOutPixels, correctOutPixels, width * height);
	printf("Error: %f\n\n", deviceErr);

	// Write results to files
	char * outFileNameBase = strtok(argv[2], "."); // Get rid of extension
	writePnm(hostOutPixels, width, height, concatStr(outFileNameBase, "_host.pnm"));
	writePnm(deviceOutPixels, width, height, concatStr(outFileNameBase, "_device.pnm"));

	// Free memories
	free(inPixels);
	free(correctOutPixels);
	free(hostOutPixels);
	free(deviceOutPixels);
	free(filter);
}
