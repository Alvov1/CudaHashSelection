#ifndef HASHSELECTION_DEVICESTORAGE_H
#define HASHSELECTION_DEVICESTORAGE_H

#include <iostream>
#include <thread>

#include "DeviceHash.h"
#include "Dictionary.h"

template <typename Char>
DEVICE size_t devStrlen(const Char *str) {
    const Char *s;
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

template <typename Char>
class DeviceStorage final {
    Char** hostArray = nullptr;
    size_t hostArraySize = 0;

    Char** deviceArray = nullptr;
    size_t* deviceArraySize = nullptr;

    Char* devicePassword = nullptr;
public:
    DeviceStorage(const std::basic_string<Char>& pass)
    : hostArraySize(Dictionary<Char>::size()), hostArray(static_cast<Char**>(malloc(Dictionary<Char>::size() * sizeof(Char*)))) {
        /* -------------- Copy an array of words to the device. -------------- */
        if(cudaSuccess != cudaMalloc(&deviceArray, hostArraySize * sizeof(char*)))
            throw std::runtime_error("DeviceStorage: CudaMalloc failed for device array."); /* Device array of pointers. */

        for(auto i = 0; i < hostArraySize; ++i) {
            size_t elemSize = strlen(Dictionary<Char>::get(i).c_str()) + 1;
            if(cudaSuccess != cudaMalloc(&hostArray[i], elemSize * sizeof(Char)))
                throw std::runtime_error("DeviceStorage: CudaMalloc failed for host array.");

            /* Allocate space for the device object, and store it on host. */
            if(cudaSuccess != cudaMemcpy(hostArray[i], Dictionary<Char>::get(i).c_str(), elemSize, cudaMemcpyHostToDevice))
                throw std::runtime_error("DeviceStorage: CudaMemcpy failed for host array."); /* Copy object to device. */
        }
        /* Copy array of host pointers to get an array of device pointers. */
        if(cudaSuccess != cudaMemcpy(deviceArray, hostArray,hostArraySize * sizeof(Char*), cudaMemcpyHostToDevice))
            throw std::runtime_error("DeviceStorage: CudaMemcpy failed for device array.");

        /* ------------- Copy the size of an array to the device. ------------ */
        if(cudaSuccess != cudaMalloc(&deviceArraySize, sizeof(size_t)))
            throw std::runtime_error("DeviceStorage: CudaMalloc failed for device array size.");
        if(cudaSuccess != cudaMemcpy(deviceArraySize, &hostArraySize, sizeof(size_t), cudaMemcpyHostToDevice))
            throw std::runtime_error("DeviceStorage: CudaMemcpy failed for device array size.");

        /* ----------------- Copy the password to the device. ---------------- */
        if(cudaSuccess != cudaMalloc(&devicePassword, sizeof(Char) * (pass.size() + 1)))
            throw std::runtime_error("DeviceStorage: CudaMalloc failed for device password.");
        if(cudaMemcpy(devicePassword, pass.c_str(), sizeof(Char) * (pass.size() + 1), cudaMemcpyHostToDevice))
            throw std::runtime_error("DeviceStorage: CudaMemcpy failed for device password.");
    }
    ~DeviceStorage() {
        for(auto i = 0; i < hostArraySize; ++i)
            if(cudaSuccess != cudaFree(hostArray[i]))
                std::cerr << "DeviceStorageage: CudaFree failed for host array." << std::endl;
        if(cudaSuccess != cudaFree(deviceArray))
            std::cerr << "DeviceStorageage: CudaFree failed for device array." << std::endl;
        free(hostArray);

        if(cudaSuccess != cudaFree(deviceArraySize))
            std::cerr << "DeviceStorageage: CudaFree failed for device array size." << std::endl;
        if(cudaSuccess != cudaFree(devicePassword))
            std::cerr << "DeviceStorageage: CudaFree failed for device password." << std::endl;
    }

    int process() {
        int copy = -1;
        int* devicePlaceForResult = nullptr;
        if(cudaSuccess != cudaMalloc(&devicePlaceForResult, sizeof(int)))
            throw std::runtime_error("DeviceStorage::process: CudaMalloc failed for device place for result.");
        if(cudaSuccess != cudaMemcpy(devicePlaceForResult, &copy, sizeof(int), cudaMemcpyHostToDevice))
            throw std::runtime_error("DeviceStorage::process: CudaMemcpy failed for device place for result.");

        const auto dimension = countDimension(Dictionary::size());
        const auto wordsPerThread = (Dictionary::size() + dimension * dimension - 1) / (dimension * dimension);
        execution<<<dimension, dimension>>>(deviceArray,
                              deviceArraySize, wordsPerThread, devicePassword, devicePlaceForResult);

        if(cudaSuccess != cudaDeviceSynchronize())
            throw std::runtime_error("DeviceStorage::process: CudaDeviceSynchronize failed.");
        if(cudaSuccess != cudaMemcpy(&copy, devicePlaceForResult, sizeof(int), cudaMemcpyDeviceToHost))
            throw std::runtime_error("DeviceStorage::process: CudaMemcpy failed for device place for result back.");
        if(cudaSuccess != cudaFree(devicePlaceForResult))
            throw std::runtime_error("DeviceStorage::process: CudaFree failed for device place for result.");
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

#endif //HASHSELECTION_DEVICESTORAGE_H
