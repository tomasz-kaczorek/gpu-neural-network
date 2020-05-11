#include "kernel.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include <math.h>
#include <stdio.h>

#define TRY(code) checkCudaErrors((cudaError_t) (code))

//copies weights to shared memory for faster access
__device__ void readWeights(float * hiddenLayerWeightsShared, float * hiddenLayerWeights, float * outputLayerWeightsShared, float * outputLayerWeights)
{
	int const id = threadIdx.x;
	for(int i = 0; i < inputLayerSize + 1; ++i)
	{
		hiddenLayerWeightsShared[i * hiddenLayerSize + id] = hiddenLayerWeights[i * hiddenLayerSize + id];
	}
	if(id < outputLayerSize)
	{
		for(int i = 0; i < hiddenLayerSize + 1; ++i)
		{
			outputLayerWeightsShared[i * outputLayerSize + id] = outputLayerWeights[i * outputLayerSize + id];
		}
	}
}

//copies weights back to global memory to store results
__device__ void writeWeights(float * hiddenLayerWeights, float * hiddenLayerWeightsShared, float * outputLayerWeights, float * outputLayerWeightsShared)
{
	int const id = threadIdx.x;
	for(int i = 0; i < inputLayerSize + 1; ++i)
	{
		hiddenLayerWeights[i * hiddenLayerSize + id] = hiddenLayerWeightsShared[i * hiddenLayerSize + id];
	}
	if(id < outputLayerSize)
	{
		for(int i = 0; i < hiddenLayerSize + 1; ++i)
		{
			outputLayerWeights[i * outputLayerSize + id] = outputLayerWeightsShared[i * outputLayerSize + id];
		}
	}
}

//copies current pattern and target
__device__ void readPattern(float * patternShared, float * patterns, int sequence)
{
	int const id = threadIdx.x;
	int index;
	for(int i = 0; i < (inputLayerSize + hiddenLayerSize - 1) / hiddenLayerSize; ++i)
	{
		index = i * hiddenLayerSize + id;
		if(index < inputLayerSize) patternShared[index] = patterns[sequence * inputLayerSize + index];
	}
}

//copies current pattern and target
__device__ void readTarget(float * targetShared, float * targets, int sequence)
{
	int const id = threadIdx.x;
	if(id < outputLayerSize)
	{
		targetShared[id] = targets[sequence * outputLayerSize + id];
	}
}

//copies current results to global memory
__device__ void writeResult(float * results, float * outputLayerOutputShared, int sequence)
{
	int const id = threadIdx.x;
	if(id < outputLayerSize)
	{
		results[sequence * outputLayerSize + id] = outputLayerOutputShared[id];
	}
}

//propagates input pattern through network
__device__ void propagate(float * hiddenLayerOutput, float * hiddenLayerWeights, float * outputLayerOutput, float * outputLayerWeights, float * pattern)
{
	int const id = threadIdx.x;
	hiddenLayerOutput[id] = 0.0f;
	for(int i = 0; i < inputLayerSize; ++i)
	{
		hiddenLayerOutput[id] += pattern[i] * hiddenLayerWeights[i * hiddenLayerSize + id];
	}
	hiddenLayerOutput[id] += -1.0f * hiddenLayerWeights[inputLayerSize * hiddenLayerSize + id];
	hiddenLayerOutput[id] = 1.0f / (1.0f + exp(-hiddenLayerOutput[id]));
	__syncthreads();
	if(id < outputLayerSize)
	{
		for(int i = 0; i < hiddenLayerSize; ++i)
		{
			outputLayerOutput[id] += hiddenLayerOutput[i] * outputLayerWeights[i * outputLayerSize + id];
		}
		outputLayerOutput[id] += -1.0f * outputLayerWeights[hiddenLayerSize * outputLayerSize + id];
		outputLayerOutput[id] = 1.0f / (1.0f + exp(-outputLayerOutput[id]));
	}
}

//calculates and backpropagates error
__device__ void backpropagate(float * hiddenLayerEror, float * outputLayerError, float * outputLayerOutput, float * outputLayerWeights, float * target)
{
	int const id = threadIdx.x;
	if(id < outputLayerSize)
	{
		outputLayerError[id] = target[id] - (outputLayerOutput[id] > 0.5f);
	}
	__syncthreads();
	hiddenLayerEror[id] = 0.0f;
	for(int i = 0; i < outputLayerSize; ++i)
	{
		hiddenLayerEror[id] += outputLayerError[i] * outputLayerWeights[id * outputLayerSize + i];
	}
}

//update weights among the network
__device__ void updateWeights(float * hiddenLayerWeights, float * hiddenLayerError, float * hiddenLayerOutput, float * outputLayerWeights, float * outputLayerError, float * outputLayerOutput, float * pattern, float * target)
{
	int const id = threadIdx.x;
	float const eta = 0.1f;
	for(int i = 0; i < inputLayerSize; ++i)
	{
		hiddenLayerWeights[i * hiddenLayerSize + id] = hiddenLayerWeights[i * hiddenLayerSize + id] + eta * hiddenLayerError[id] * hiddenLayerOutput[id] * (1.0f - hiddenLayerOutput[id]) * pattern[i];
	}
	hiddenLayerWeights[inputLayerSize * hiddenLayerSize + id] = hiddenLayerWeights[inputLayerSize * hiddenLayerSize + id] + eta * hiddenLayerError[id] * hiddenLayerOutput[id] * (1.0f - hiddenLayerOutput[id]) * (-1.0f);
	__syncthreads();
	if(id < outputLayerSize)
	{
		for(int i = 0; i < hiddenLayerSize; ++i)
		{
			outputLayerWeights[i * outputLayerSize + id] = outputLayerWeights[i * outputLayerSize + id] + eta * outputLayerError[id] * outputLayerOutput[id] * (1.0f - outputLayerOutput[id]) * hiddenLayerOutput[i];
		}
		outputLayerWeights[hiddenLayerSize * outputLayerSize + id] = outputLayerWeights[hiddenLayerSize * outputLayerSize + id] + eta * outputLayerError[id] * outputLayerOutput[id] * (1.0f - outputLayerOutput[id]) * (-1.0f);
	}
}

//cuda kernel - runs through single epoch
__global__ void kernel(float * hiddenLayerWeights, float * outputLayerWeights, float * patterns, float * targets, float * results, int numberOfPatterns)
{
	__shared__ float hiddenLayerWeightsShared[(inputLayerSize + 1) * hiddenLayerSize];
	__shared__ float hiddenLayerOutputShared[hiddenLayerSize];
	__shared__ float patternShared[inputLayerSize];
	__shared__ float targetShared[outputLayerSize];
	__shared__ float hiddenLayerErrorShared[hiddenLayerSize];
    __shared__ float outputLayerWeightsShared[(hiddenLayerSize + 1) * outputLayerSize];
	__shared__ float outputLayerOutputShared[outputLayerSize];
	__shared__ float outputLayerErrorShared[outputLayerSize];

	readWeights(hiddenLayerWeightsShared, hiddenLayerWeights, outputLayerWeightsShared, outputLayerWeights);
	for(int i = 0; i < numberOfPatterns; ++i)
	{
		readPattern(patternShared, patterns, i);
	__syncthreads();
		readTarget(targetShared, targets, i);
	__syncthreads();
		propagate(hiddenLayerOutputShared, hiddenLayerWeightsShared, outputLayerOutputShared, outputLayerWeightsShared, patternShared);
	__syncthreads();
		backpropagate(hiddenLayerErrorShared, outputLayerErrorShared, outputLayerOutputShared, outputLayerWeightsShared, targetShared);
	__syncthreads();
		updateWeights(hiddenLayerWeightsShared, hiddenLayerErrorShared, hiddenLayerOutputShared, outputLayerWeightsShared, outputLayerErrorShared, outputLayerOutputShared, patternShared, targetShared);
	__syncthreads();
		writeResult(results, outputLayerOutputShared, i);
	}
	writeWeights(hiddenLayerWeights, hiddenLayerWeightsShared, outputLayerWeights, outputLayerWeightsShared);
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t train(float * hiddenLayerWeights, float * outputLayerWeights, float * patterns, float * targets)
{
    float * hiddenLayerWeightsGlobal;
    float * outputLayerWeightsGlobal;
	float * patternsGlobal;
	float * targetsGlobal;
	float * resultsGlobal;

	float * results = new float[numberOfPatterns * outputLayerSize];

    // Choose which GPU to run on, change this on a multi-GPU system.
    TRY(cudaSetDevice(0));

    TRY(cudaMalloc((void**)&hiddenLayerWeightsGlobal, (inputLayerSize + 1) * hiddenLayerSize * sizeof(float)));
    TRY(cudaMalloc((void**)&outputLayerWeightsGlobal, (hiddenLayerSize + 1) * outputLayerSize * sizeof(float)));
    TRY(cudaMalloc((void**)&patternsGlobal, numberOfPatterns * inputLayerSize * sizeof(float)));
    TRY(cudaMalloc((void**)&targetsGlobal, numberOfPatterns * outputLayerSize * sizeof(float)));
    TRY(cudaMalloc((void**)&resultsGlobal, numberOfPatterns * outputLayerSize * sizeof(float)));

    // Copy input vectors from host memory to GPU buffers.
    TRY(cudaMemcpy(hiddenLayerWeightsGlobal, hiddenLayerWeights, (inputLayerSize + 1) * hiddenLayerSize * sizeof(float), cudaMemcpyHostToDevice));
    TRY(cudaMemcpy(outputLayerWeightsGlobal, outputLayerWeights, (hiddenLayerSize + 1) * outputLayerSize * sizeof(float), cudaMemcpyHostToDevice));
    TRY(cudaMemcpy(patternsGlobal, patterns, numberOfPatterns * inputLayerSize * sizeof(float), cudaMemcpyHostToDevice));
    TRY(cudaMemcpy(targetsGlobal, targets, numberOfPatterns * outputLayerSize * sizeof(float), cudaMemcpyHostToDevice));

    // Launch a kernel on the GPU with one thread for each element.
	for(int i = 0; i < 10000; ++i)
	{
		kernel<<<1, hiddenLayerSize>>>(hiddenLayerWeightsGlobal, outputLayerWeightsGlobal, patternsGlobal, targetsGlobal, resultsGlobal, numberOfPatterns);
		TRY(cudaMemcpy(results, resultsGlobal, numberOfPatterns * outputLayerSize * sizeof(float), cudaMemcpyDeviceToHost));
		float MSE = 0.0;
		bool OK = true;
		int correct = 0;
		for(int j = 0; j < numberOfPatterns; ++j)
		{
			for(int k = 0; k < outputLayerSize; ++k)
			{
				//printf("Epoch %3d pattern %-3d %3d %12f\n", i, j, (int) targets[j * outputLayerSize + k], results[j * outputLayerSize + k]);
				MSE += pow(targets[j * outputLayerSize + k] - results[j * outputLayerSize + k], 2) / (outputLayerSize * numberOfPatterns);
				if(results[j * outputLayerSize + k] < 0.5 && targets[j * outputLayerSize + k] == 1 || results[j * outputLayerSize + k] >= 0.5 && targets[j * outputLayerSize + k] == 0) OK = false;
			}
			if(OK) ++correct;
			OK = true;
			//printf("\n");
		}
		//printf("Epoch %3d MSE %f\n\n", i, MSE);
		if(i % 100 == 0) printf("Epoch %3d # of correct %3d MSE %f\n\n", i, correct, MSE);
	}

    // Check for any errors launching the kernel
    TRY(cudaGetLastError());
    
    //cudaDeviceSynchronize waits for the kernel to finish, and returns
    //any errors encountered during the launch.
    TRY(cudaDeviceSynchronize());

    cudaFree(hiddenLayerWeightsGlobal);
    cudaFree(outputLayerWeightsGlobal);
	cudaFree(patternsGlobal);
	cudaFree(targetsGlobal);

    return (cudaError_t) 0; //cudaStatus;
}
