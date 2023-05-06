#include <iostream>

#include "HostHash.h"
#include "DeviceHash.h"
#include "TimeLogger.h"
#include "HashSelection.h"

int main() {
    const std::filesystem::path dictionaryLocation("../../Dictionaries/100.txt");
    const auto words = HashSelection::readFileDictionary(dictionaryLocation);
    Time::cout << "Loaded dictionary. Complexity: " << HashSelection::countComplexity(words) << Time::endl;

    const HostSHA256 hash = [&words] {
        const HashSelection::Word mutation = HashSelection::getRandomModification(words);
        return HostSHA256 { mutation.data, mutation.size * sizeof(HashSelection::Char) };
    } ();
    Time::cout << "Chosen word with hash " << hash.to_string() << Time::endl;

    HashSelection::runDevice(words, hash);

    return 0;
}