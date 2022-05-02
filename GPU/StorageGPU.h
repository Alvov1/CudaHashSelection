#ifndef HASHSELECTION_STORAGEGPU_H
#define HASHSELECTION_STORAGEGPU_H

#include <iostream>
#include <thread>
#include "DeviceHash.h"

DEVICE size_t devStrlen(const char *str) {
    const char *s;
    for (s = str; *s; ++s);
    return (s - str);
}

__global__ void execution(const char* const* deviceArray, const size_t* arraySize,
                          unsigned wordsPerThread, const char* password, int* result) {
    const auto threadNumber = blockIdx.x * blockDim.x + threadIdx.x;
    auto arrayPosition = threadNumber * wordsPerThread;
    const auto wordsToCheck = (arrayPosition + wordsPerThread < *arraySize) ?
            wordsPerThread : *arraySize - arrayPosition;

    for(auto i = 0; i < wordsToCheck; ++arrayPosition, ++i) {
        if(*result != -1) break;
        const char* currentWord = deviceArray[arrayPosition];

        DeviceSHA256 sha(currentWord, devStrlen(currentWord));
        const auto hash = sha.count();
        if(hash.compare(password)) {
            printf("Found at %d\n", arrayPosition);
            auto old = atomicExch(result, static_cast<int>(arrayPosition));
            printf("Old: %d\n", old);
            printf("Value now: %d\n", *result);
        }
    }
}

class StorageGPU final {
    char** hostArray = nullptr;
    size_t hostArraySize = 0;

    char** deviceArray = nullptr;
    size_t* deviceArraySize = nullptr;

    char* devicePassword = nullptr;
public:
    StorageGPU(const char* const* const hostArray_, size_t size_, const std::string& pass)
    : hostArraySize(size_), hostArray(static_cast<char**>(malloc(size_ * sizeof(char*)))) {
        /* -------------- Copy an array of words to the device. -------------- */
        cudaMalloc(&deviceArray, hostArraySize * sizeof(char*)); /* Device array of pointers. */
        for(auto i = 0; i < hostArraySize; ++i) {
            size_t elemSize = strlen(hostArray_[i]) + 1;
            cudaMalloc(&hostArray[i], elemSize * sizeof(char));
                /* Allocate space for the device object, and store it on host. */
            cudaMemcpy(hostArray[i], hostArray_[i], elemSize,
                       cudaMemcpyHostToDevice); /* Copy object to device. */
        }
        cudaMemcpy(deviceArray, hostArray, hostArraySize * sizeof(char*),
                   cudaMemcpyHostToDevice); /* Copy array of host pointers to get an array of device pointers. */

        /* ------------- Copy the size of an array to the device. ------------ */
        cudaMalloc(&deviceArraySize, sizeof(size_t));
        cudaMemcpy(deviceArraySize, &hostArraySize, sizeof(size_t), cudaMemcpyHostToDevice);

        /* ----------------- Copy the password to the device. ---------------- */
        cudaMalloc(&devicePassword, sizeof(char) * (pass.size() + 1));
        cudaMemcpy(devicePassword, pass.c_str(), sizeof(char) * (pass.size() + 1), cudaMemcpyHostToDevice);
    }
    ~StorageGPU() {
        for(auto i = 0; i < hostArraySize; ++i)
            cudaFree(hostArray[i]);
        cudaFree(deviceArray);
        free(hostArray);

        cudaFree(deviceArraySize);
        cudaFree(devicePassword);
    }


    int process() {
        int copy = -1;
        int* devicePlaceForResult = nullptr;
        cudaMalloc(&devicePlaceForResult, sizeof(int));
        cudaMemcpy(devicePlaceForResult, &copy, sizeof(int), cudaMemcpyHostToDevice);

        execution<<<16, 16>>>(deviceArray,
                              deviceArraySize, 12, devicePassword, devicePlaceForResult);

        cudaDeviceSynchronize();
        cudaMemcpy(&copy, devicePlaceForResult, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(devicePlaceForResult);
        return copy;
    }
};

#endif //HASHSELECTION_STORAGEGPU_H
