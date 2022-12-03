#include <functional>
#include "Timer.h"
#include "HostHash.h"
#include "PasswordDictionary.h"

#include "DevicePointer.h"
#include "DeviceFunctions.h"
#include "DeviceDictionary.h"

unsigned devicesAvailable() {
    int deviceCount = 0;
    if(cudaSuccess != cudaGetDeviceCount(&deviceCount))
        throw std::runtime_error("CudaGetDeviceCount failed.");
    if(deviceCount == 0)
        throw std::runtime_error("There are no any Cuda-capable devices available.");
    return static_cast<unsigned>(deviceCount);
}

int main() {
    Console::cout << "Found " << devicesAvailable() << " cuda devices available." << Console::endl;

    ReplacementDictionary mutations;
    mutations.show();

    PasswordDictionary passwords;
    passwords.calculateQuantities(mutations);

    const auto& plainPassword = passwords.getRandom();
    Console::timer << "Using word: " << plainPassword << Console::endl;
    const auto mutatedPassword = mutations.rearrange(plainPassword);
    Console::timer << "Using after rearrangement: " << mutatedPassword << Console::endl;
    const HostSHA256 hash(mutatedPassword.c_str(), mutatedPassword.size());
    Console::timer << "Searching for mutatedPassword with hash '" << hash.to_string() << "'." << Console::endl;

    const DeviceDictionary devicePasswords(passwords.get());
    const DeviceDictionary deviceMutations(mutations.get());
    const DevicePointer password(hash.to_string().c_str(), hash.to_string().size());

    return 0;
}