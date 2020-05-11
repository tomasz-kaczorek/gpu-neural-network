#include "cuda_runtime.h"

int const inputLayerSize = 54; //54
int const hiddenLayerSize = 150; //32
int const outputLayerSize = 8; //8
int const numberOfPatterns = 180; //70 for regular, 180 for exhaustive

cudaError_t train(float * hiddenLayerWeights, float * outputLayerWeights, float * patterns, float * targets);