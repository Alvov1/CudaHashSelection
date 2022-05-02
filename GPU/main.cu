#include <iostream>

#include "Dictionary.h"
#include "StorageGPU.h"
#include "HostHash.h"
#include "DeviceHash.h"
#include "Timer.h"

int main() {
    const auto pass = Dictionary::giveRandom();
    const auto hostHash = HostSHA256(pass).out();
    Timer::out << "Searching for password with hash '" << hostHash << "'." << std::endl;

    StorageGPU storage(Dictionary::data(), Dictionary::size(), hostHash);
    Timer::out << "Words are loaded on device." << std::endl;

    const auto result = storage.process();
    Timer::out << "Founded coincidence on place " << result << "." << std::endl;

    return 0;
}