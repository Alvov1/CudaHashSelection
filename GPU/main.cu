#include <iostream>

#include "HostHash.h"
#include "DeviceHash.h"
#include "TimeLogger.h"
#include "HashSelection.h"

int main() {
    const std::filesystem::path dictionaryLocation("../../Dictionaries/100.txt");
    const auto words = HashSelection::readFileDictionary(dictionaryLocation);
    Time::cout << "Loaded dictionary. Complexity: " << HashSelection::countComplexity(words) << Time::endl;

    const HashSelection::Word mutation = HashSelection::getRandomModification(words);
    HashSelection::Closure closure = [&mutation] (const HashSelection::Word& forWord) {
        static const HostSHA256 hash { mutation.data, mutation.size * sizeof(HashSelection::Char) };
        const HostSHA256 another(forWord.data, forWord.size * sizeof(HashSelection::Char));
        return std::memcmp(hash.get().data(), another.get().data(), 32) == 0;
    };
    Time::cout << "Chosen word: " << mutation << Time::endl;

    HashSelection::runDevice(words, closure);

    return 0;
}