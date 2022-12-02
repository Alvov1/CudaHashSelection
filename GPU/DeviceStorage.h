#ifndef HASHSELECTION_DEVICESTORAGE_H
#define HASHSELECTION_DEVICESTORAGE_H

#include <iostream>
#include <thread>

#include "DeviceHash.h"
#include "PasswordDictionary.h"
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
    DeviceStorage(const std::basic_string<Char>& pass);
    ~DeviceStorage();

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
