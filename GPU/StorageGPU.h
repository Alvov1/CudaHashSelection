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
    if(arrayPosition >= *arraySize) return;
    const auto wordsToCheck = (arrayPosition + wordsPerThread < *arraySize) ?
            wordsPerThread : *arraySize - arrayPosition;

    for(auto i = 0; i < wordsToCheck; ++arrayPosition, ++i) {
        if(*result != -1) break;

        const char* currentWord = deviceArray[arrayPosition];
        DeviceSHA256 sha(currentWord, devStrlen(currentWord));
        const auto hash = sha.count();
        if(hash.equals(password))
            auto old = atomicExch(result, static_cast<int>(arrayPosition));
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
        if(cudaSuccess != cudaMalloc(&deviceArray, hostArraySize * sizeof(char*)))
            throw std::runtime_error("StorageGPU: CudaMalloc failed for device array."); /* Device array of pointers. */

        for(auto i = 0; i < hostArraySize; ++i) {
            size_t elemSize = strlen(hostArray_[i]) + 1;
            if(cudaSuccess != cudaMalloc(&hostArray[i], elemSize * sizeof(char)))
                throw std::runtime_error("StorageGPU: CudaMalloc failed for host array.");

            /* Allocate space for the device object, and store it on host. */
            if(cudaSuccess != cudaMemcpy(hostArray[i], hostArray_[i], elemSize, cudaMemcpyHostToDevice))
                throw std::runtime_error("StorageGPU: CudaMemcpy failed for host array."); /* Copy object to device. */
        }
        if(cudaSuccess != cudaMemcpy(deviceArray, hostArray,
        hostArraySize * sizeof(char*), cudaMemcpyHostToDevice))
            throw std::runtime_error("StorageGPU: CudaMemcpy failed for device array."); /* Copy array of host pointers to get an array of device pointers. */

        /* ------------- Copy the size of an array to the device. ------------ */
        if(cudaSuccess != cudaMalloc(&deviceArraySize, sizeof(size_t)))
            throw std::runtime_error("StorageGPU: CudaMalloc failed for device array size.");
        if(cudaSuccess != cudaMemcpy(deviceArraySize, &hostArraySize, sizeof(size_t), cudaMemcpyHostToDevice))
            throw std::runtime_error("StorageGPU: CudaMemcpy failed for device array size.");

        /* ----------------- Copy the password to the device. ---------------- */
        if(cudaSuccess != cudaMalloc(&devicePassword, sizeof(char) * (pass.size() + 1)))
            throw std::runtime_error("StorageGPU: CudaMalloc failed for device password.");
        if(cudaMemcpy(devicePassword, pass.c_str(), sizeof(char) * (pass.size() + 1), cudaMemcpyHostToDevice))
            throw std::runtime_error("StorageGPU: CudaMemcpy failed for device password.");
    }
    ~StorageGPU() {
        for(auto i = 0; i < hostArraySize; ++i)
            if(cudaSuccess != cudaFree(hostArray[i]))
                std::cerr << "~StorageGPU: CudaFree failed for host array." << std::endl;
        if(cudaSuccess != cudaFree(deviceArray))
            std::cerr << "~StorageGPU: CudaFree failed for device array." << std::endl;
        free(hostArray);

        if(cudaSuccess != cudaFree(deviceArraySize))
            std::cerr << "~StorageGPU: CudaFree failed for device array size." << std::endl;
        if(cudaSuccess != cudaFree(devicePassword))
            std::cerr << "~StorageGPU: CudaFree failed for device password." << std::endl;
    }

    int process() {
        int copy = -1;
        int* devicePlaceForResult = nullptr;
        if(cudaSuccess != cudaMalloc(&devicePlaceForResult, sizeof(int)))
            throw std::runtime_error("StorageGPU::process: CudaMalloc failed for device place for result.");
        if(cudaSuccess != cudaMemcpy(devicePlaceForResult, &copy, sizeof(int), cudaMemcpyHostToDevice))
            throw std::runtime_error("StorageGPU::process: CudaMemcpy failed for device place for result.");

        const auto dimension = countDimension(Dictionary::size());
        const auto wordsPerThread = (Dictionary::size() + dimension * dimension - 1) / (dimension * dimension);
        execution<<<dimension, dimension>>>(deviceArray,
                              deviceArraySize, wordsPerThread, devicePassword, devicePlaceForResult);

        cudaError_t error = cudaDeviceSynchronize();
        if(error != cudaSuccess) {
            std::cerr << cudaGetErrorString(error) << std::endl;
            throw std::runtime_error("StorageGPU::process: CudaDeviceSynchronize failed.");
        }

        if(cudaSuccess != cudaMemcpy(&copy, devicePlaceForResult, sizeof(int), cudaMemcpyDeviceToHost))
            throw std::runtime_error("StorageGPU::process: CudaMemcpy failed for device place for result back.");
        if(cudaSuccess != cudaFree(devicePlaceForResult))
            throw std::runtime_error("StorageGPU::process: CudaFree failed for device place for result.");
        return copy;
    }

    static unsigned countDimension(size_t dictionarySize) {
        auto base = static_cast<unsigned>(std::cbrt(dictionarySize));

        for(unsigned power = 2; ; ++power) {
            const auto value = static_cast<unsigned>(pow(2, power + 1));
            if(value > base) return static_cast<unsigned>(value / 2);
        }
    }
};

#endif //HASHSELECTION_STORAGEGPU_H
