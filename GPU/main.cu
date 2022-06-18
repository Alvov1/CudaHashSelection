#include <iostream>

#include "Dictionary.h"
#include "StorageGPU.h"
#include "HostHash.h"
#include "Timer.h"

unsigned devicesAvailable() {
    int deviceCount = 0;
    if(cudaSuccess != cudaGetDeviceCount(&deviceCount))
        throw std::runtime_error("CudaGetDeviceCount failed.");
    if(deviceCount == 0)
        throw std::runtime_error("There are no any Cuda-capable devices available.");
    return static_cast<unsigned>(deviceCount);
}

int main() {
    Timer::out << "Founded " << devicesAvailable() << " Cuda-capable devices available." << std::endl;

    const auto pass = Dictionary::giveRandom();
    const auto hostHash = HostSHA256(pass).out();
    Timer::out << "Searching for password with hash '" << hostHash << "'." << std::endl;

    StorageGPU storage(Dictionary::data(), Dictionary::size(), hostHash);
    Timer::out << "Words are loaded on device." << std::endl;

    const auto result = storage.process();
    if(result != -1)
        Timer::out << "Fount coincidence on place " << result << ". Password is '" << Dictionary::instance()[result] << "'." << std::endl;

    return 0;
}