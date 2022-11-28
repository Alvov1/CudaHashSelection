#include <iostream>

#include "ScreenWriter.h"
#include "Dictionary.h"
#include "HostHash.h"
#include "DeviceStorage.h"

unsigned devicesAvailable() {
    int deviceCount = 0;
    if(cudaSuccess != cudaGetDeviceCount(&deviceCount))
        throw std::runtime_error("CudaGetDeviceCount failed.");
    if(deviceCount == 0)
        throw std::runtime_error("There are no any Cuda-capable devices available.");
    return static_cast<unsigned>(deviceCount);
}

int main() {
    using Char = char;
    ReplacementDictionary<Char>::showVariants();
    Dictionary<Char>::calculateQuantities();

    const auto plainPassword = Dictionary<Char>::getRandom();
    Console::timer << "Using word: " << plainPassword << Console::endl;

    const auto mutatedPassword = ReplacementDictionary<Char>::rearrange(plainPassword);
    Console::timer << "Using after rearrangement: " << mutatedPassword << Console::endl;

    const HostSHA256 hash(mutatedPassword.c_str(), mutatedPassword.size());
    Console::timer << "Searching for mutatedPassword with hash '" << hash.to_string() << "'." << Console::endl;

    Dictionary<Char>::find(hash.to_string());
    return 0;
}