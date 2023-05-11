#include <iostream>

#include "Utility.h"
#include "HostHash.h"
#include "DeviceHash.h"
#include "TimeLogger.h"
#include "HashSelectionDevice.h"

int main() {
    const std::filesystem::path dictionaryLocation("../../Dictionaries/10000-Truncated.txt");
    const auto words = HashSelection::readFileDictionary(dictionaryLocation);
    Time::cout << "Loaded dictionary of " << words.size() << " words. Total combinations count: "
        << HashSelection::countComplexity(words, false) << Time::endl;

    const Hash::HostSHA256 hash = [&words] {
        HashSelection::Word mutation = HashSelection::getRandomModification(words);
        Hash::HostSHA256 hash { mutation.first, mutation.second * sizeof(HashSelection::Char) };
        Time::cout << "Chosen word " << mutation << " with hash " << hash.to_string() << Time::endl;
        return hash;
    } ();

    const auto value = HashSelection::process(words, hash);
    if(value.has_value()) Time::cout << "Completed: " << *value << Time::endl;
        else Time::cout << "Not found." << Time::endl;

    return 0;
}