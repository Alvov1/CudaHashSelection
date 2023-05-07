#include <iostream>

#include "Utility.h"
#include "HostHash.h"
#include "DeviceHash.h"
#include "TimeLogger.h"
#include "HashSelectionDevice.h"

int main() {
    const std::filesystem::path dictionaryLocation("../../Dictionaries/128.txt");
    const auto words = HashSelection::readFileDictionary(dictionaryLocation);
    Time::cout << "Loaded dictionary. Complexity: " << HashSelection::countComplexity(words) << Time::endl;

    const Hash::HostSHA256 hash = [&words] {
        const HashSelection::Word mutation = HashSelection::getRandomModification(words);
        Hash::HostSHA256 hash { mutation.first, mutation.second * sizeof(HashSelection::Char) };
        Time::cout << "Chosen word " << mutation << " with hash " << hash.to_string() << Time::endl;
        return hash;
    } ();

    const auto value = HashSelection::runDevice(words, hash);
    if(value != nullptr) Time::cout << "Completed: " << *value << Time::endl;
        else Time::cout << "Not found." << Time::endl;

    return 0;
}