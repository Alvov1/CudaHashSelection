#ifndef HASHSELECTION_DEVICESTORAGE_H
#define HASHSELECTION_DEVICESTORAGE_H

#include <iostream>
#include <thread>

#include "DeviceHash.h"
#include "Dictionary.h"
#include "ReplacementDictionary.h"

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
    /* Used for the dictionary. */
    Char** hostDictionaryPointersArray = nullptr;
    size_t hostDictionaryPointersArraySize = 0;

    Char** deviceDictionaryPointersArray = nullptr;
    size_t* deviceDictionaryPointersArraySize = nullptr;

    /* Used for the replacements array. There is no need for the size variables as
     * the number of English letters does not change over time*/
    static constexpr auto englishLettersNumber = 26;
    Char** hostVariantsPointersArray = nullptr;
    Char** deviceVariantsPointersArray = nullptr;

    /* Hash that we are looking for. */
    Char* devicePassword = nullptr;
public:
    DeviceStorage(const std::basic_string<Char>& pass) {
        /* --------------------- Copying the dictionary. --------------------- */
        /*
         * 1. Allocate space for array of pointers in RAM.
         * 2. In each pointer locate a word from the dictionary in the GPU memory.
         * 3. Copy array of pointers from RAM to GPU
        */

        // Point 1:
        hostDictionaryPointersArraySize = Dictionary<Char>::size();
        hostDictionaryPointersArray = static_cast<Char**>(malloc(Dictionary<Char>::size() * sizeof(Char*)));
        if(hostDictionaryPointersArray == nullptr)
            throw std::runtime_error("DeviceStorage: malloc failed for the dictionary host array.");

        // Point 2:
        for(auto i = 0; i < hostDictionaryPointersArraySize; ++i) {
            size_t elemSize = Dictionary<Char>::get(i).size() + 1;
            if(cudaSuccess != cudaMalloc(&hostDictionaryPointersArray[i], elemSize * sizeof(Char)))
                throw std::runtime_error("DeviceStorage: CudaMalloc failed for host array.");
            if(cudaSuccess != cudaMemcpy(hostDictionaryPointersArray[i], Dictionary<Char>::get(i).c_str(), elemSize, cudaMemcpyHostToDevice))
                throw std::runtime_error("DeviceStorage: CudaMemcpy failed for host array.");
        }

        // Point 3:
        if(cudaSuccess != cudaMalloc(&deviceDictionaryPointersArray, hostDictionaryPointersArraySize * sizeof(Char*)))
            throw std::runtime_error("DeviceStorage: CudaMalloc failed for device array.");
        if(cudaSuccess != cudaMemcpy(deviceDictionaryPointersArray, hostDictionaryPointersArray, hostDictionaryPointersArraySize * sizeof(Char*), cudaMemcpyHostToDevice))
            throw std::runtime_error("DeviceStorage: CudaMemcpy failed for device array.");
        
        /* ----------- Copying the size of an array to the device. ----------- */
        if(cudaSuccess != cudaMalloc(&deviceDictionaryPointersArraySize, sizeof(size_t)))
            throw std::runtime_error("DeviceStorage: CudaMalloc failed for device array size.");
        if(cudaSuccess != cudaMemcpy(deviceDictionaryPointersArraySize, &hostDictionaryPointersArraySize, sizeof(size_t), cudaMemcpyHostToDevice))
            throw std::runtime_error("DeviceStorage: CudaMemcpy failed for device array size.");

        /* --------------- Copying the password to the device. --------------- */
        hostVariantsPointersArray = static_cast<Char**>(malloc(englishLettersNumber * sizeof(Char*)));
        if(hostVariantsPointersArray == nullptr)
            throw std::runtime_error("DeviceStorage: malloc failed for the variants host array.");

        for(unsigned i = 0; i < englishLettersNumber; ++i) {
            size_t elemSize = Rep
        }
        
        /* --------------- Copying the password to the device. --------------- */
        if(cudaSuccess != cudaMalloc(&devicePassword, sizeof(Char) * (pass.size() + 1)))
            throw std::runtime_error("DeviceStorage: CudaMalloc failed for device password.");
        if(cudaMemcpy(devicePassword, pass.c_str(), sizeof(Char) * (pass.size() + 1), cudaMemcpyHostToDevice))
            throw std::runtime_error("DeviceStorage: CudaMemcpy failed for device password.");
    }
    ~DeviceStorage() {
        for(auto i = 0; i < hostDictionaryPointersArraySize; ++i)
            if(cudaSuccess != cudaFree(hostDictionaryPointersArray[i]))
                std::cerr << "DeviceStorageage: CudaFree failed for host array." << std::endl;
        if(cudaSuccess != cudaFree(deviceDictionaryPointersArray))
            std::cerr << "DeviceStorageage: CudaFree failed for device array." << std::endl;
        free(hostDictionaryPointersArray);

        if(cudaSuccess != cudaFree(deviceDictionaryPointersArraySize))
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
        execution<<<dimension, dimension>>>(deviceDictionaryPointersArray,
                              deviceDictionaryPointersArraySize, wordsPerThread, devicePassword, devicePlaceForResult);

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
