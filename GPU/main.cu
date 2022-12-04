#include "Timer.h"
#include "HostHash.h"
#include "DeviceFunctions.h"
#include "CudaSpecification.h"
#include "DictionaryOnDevice.h"
#include "PasswordDictionary.h"

int main() {
    Hardware::checkCudaDevices();

    MutationDictionary mutations;
    mutations.show();
    PasswordDictionary passwords;
    passwords.calculateQuantities(mutations);

//    const auto& plainPassword = passwords.getRandom();
//    Console::timer << "Using word: " << plainPassword << Console::endl;
//    const auto mutatedPassword = mutations.mutate(plainPassword);
//    Console::timer << "Using after rearrangement: " << mutatedPassword << Console::endl;
    std::string mutatedPassword = "hello";
    const HostSHA256 hash(mutatedPassword.c_str(), mutatedPassword.size());
    Console::timer << "Searching for requiredHash with hash '" << hash.to_string() << "'." << Console::endl;

    const DictionaryOnDevice devicePasswords(passwords.get());
    const DictionaryOnDevice deviceMutations(mutations.get());
    const ArrayOnDevice requiredHash(hash.to_string().c_str(), hash.to_string().size());
    const ArrayOnDevice placeForResult(20);

    Console::timer << "Data is loaded to device." << Console::endl;

    const auto dimension = PasswordDictionary::upperPower2(passwords.size());
    DeviceFunctions::process<<<dimension, 2>>>(
            devicePasswords.getArray(),
            devicePasswords.size(),
            deviceMutations.getArray(),
            1,
            requiredHash.get(),
            placeForResult.get());
    cudaDeviceSynchronize();

    const auto result = placeForResult.readBack();
    if(result != std::string(20, '0'))
        Console::timer << "Found coincidence with word " << result << "." << Console::endl;

    return 0;
}