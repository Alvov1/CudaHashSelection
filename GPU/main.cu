#include <iostream>

#include "HostHash.h"
#include "DeviceHash.h"
#include "TimeLogger.h"
#include "HashSelectionDevice.h"
#include "../CPU/Utility.h"

int main() {
//    const std::filesystem::path dictionaryLocation("../../Dictionaries/128.txt");
//    const auto words = HashSelection::readFileDictionary(dictionaryLocation);
//    Time::cout << "Loaded dictionary. Complexity: " << HashSelection::countComplexity(words) << Time::endl;
//
//    const Hash::HostSHA256 hash = [&words] {
//        const HashSelection::Word mutation = HashSelection::getRandomModification(words);
//        return Hash::HostSHA256 { mutation.data, mutation.size * sizeof(HashSelection::Char) };
//    } ();
//    Time::cout << "Chosen word with hash " << hash.to_string() << Time::endl;
//
//    const auto value = HashSelection::runDevice(words, hash);
//    if(value.has_value()) Time::cout << "Completed: " << *value << Time::endl;
//        else Time::cout << "Not found." << Time::endl;

    HashSelection::test<<<4, 4>>>();
    if(cudaSuccess != cudaDeviceSynchronize())
        std::cerr << "Failed." << std::endl;

    return 0;
}