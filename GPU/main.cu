#include <iostream>

#include "DeviceFunctions.h"
#include "Dictionary.h"
#include "DeviceHash.h"
#include "HostHash.h"
#include "Timer.h"

__global__ void Process(const char* hash, size_t arraySize, size_t wordsPerThread, int* result) {
    const auto threadNumber = blockIdx.x * blockDim.x + threadIdx.x;
    auto arrayPosition = threadNumber * wordsPerThread;
    if(arrayPosition > arraySize) return;
    const auto wordsToCheck = (arrayPosition + wordsPerThread < arraySize) ?
            wordsPerThread : arraySize - arrayPosition;

    for(auto i = 0; i < wordsToCheck; ++arrayPosition, ++i) {
        if(*result != -1) return;
        const char* currentWord = DeviceFunctions::word(arrayPosition);

        DeviceSHA256 sha(currentWord, DeviceFunctions::strlen(currentWord));
        const auto value = sha.count();

        if(DeviceFunctions::strcmp(value.get(), hash) == 0) {
            printf("Found coincidence at %llu. Word is '%s'.\n", arrayPosition, currentWord);
            atomicAdd(result, static_cast<int>(arrayPosition));
            break;
        }
    }
}

int main() {
    const auto password = Dictionary::giveRandom();
    const auto hash = HostSHA256(password).out();
    Timer::out << "Chosen word '" << password << "' with hash '" << hash << "'." << std::endl;

    char* deviceHash = nullptr;
    cudaMalloc(&deviceHash, sizeof(char) * (hash.size() + 1));
    cudaMemcpy(deviceHash, hash.c_str(), sizeof(char) * (hash.size() + 1), cudaMemcpyHostToDevice);
    int* result = nullptr; int copy = -1;
    cudaMalloc(&result, sizeof(int));
    cudaMemcpy(result, &copy, sizeof(int), cudaMemcpyHostToDevice);

    Timer::out << "Loaded hash to the device. Starting researching process." << std::endl;

    Process<<<16, 16>>>(deviceHash, 2997, 12, result);
    cudaDeviceSynchronize();

    Timer::out << "Process ended." << std::endl;
    return 0;
}