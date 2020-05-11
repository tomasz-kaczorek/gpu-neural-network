#include "kernel.h"

#include <stdio.h>
#include <stdlib.h>

void readData(float * patterns, float * targets)
{
	FILE * file = fopen("rs_train5_exhaustive_no_headers.txt", "r");
	if(file != NULL)
	{
		float read;
		signed char temp;
		for(int i = 0; i < numberOfPatterns; ++i)
		{
			for(int j = 0; j < inputLayerSize; ++j)
			{
				fscanf(file, "%f", &patterns[i * inputLayerSize + j]);
			}
			fscanf(file, "%f", &read);
			temp = (signed char) read;
			for(int j = 0; j < outputLayerSize; ++j)
			{
				targets[i * outputLayerSize + j] = (float) (temp >> j & 1);
			}
		}
	}
	else printf("Error occured during data read.");
}

void readWeights(float * hiddenLayerWeights, float * outputLayerWeights)
{
	for(int i = 0; i < (inputLayerSize + 1) * hiddenLayerSize; ++i)
	{
		hiddenLayerWeights[i] = 0.0f;
	}
	for(int i = 0; i < (hiddenLayerSize + 1) * outputLayerSize; ++i)
	{
		outputLayerWeights[i] = 0.0f;
	}
}

int main()
{
	float * patterns = new float[numberOfPatterns * inputLayerSize];
	float * targets = new float[numberOfPatterns * outputLayerSize];

	float * hiddenLayerWeights = new float[(inputLayerSize + 1) * hiddenLayerSize];
	float * outputLayerWeights = new float[(hiddenLayerSize + 1) * outputLayerSize];

	_sleep(2000);

	readData(patterns, targets);
	readWeights(hiddenLayerWeights, outputLayerWeights);
	
	
	train(hiddenLayerWeights, outputLayerWeights, patterns, targets);

	printf("Done.\n");

	getchar();

    return 0;
}