#include <iostream>

#include "HostHash.h"
#include "TimeLogger.h"
#include "HashSelectionDevice.h"

int main(int argc, const char* const * argv) {
    const std::filesystem::path dictionaryLocation = (argc > 1) ? std::filesystem::path { argv[1] }
        : std::filesystem::path { "Dictionary.txt" };

    try {
        const auto words = HashSelection::readFileDictionary(dictionaryLocation);
        Time::cout << "Loaded dictionary of " << words.size() << " words." << Time::endl;
        Time::cout << "Total combinations count: " << HashSelection::countComplexity(words, false) << Time::endl;

        const Hash::HostSHA256 hash = [&words] {
            HashSelection::Word mutation = HashSelection::getRandomModification(words);
            Hash::HostSHA256 hash{mutation.first, mutation.second * sizeof(HashSelection::Char)};
            Time::cout << "Chosen word " << mutation << " with hash " << hash.to_string() << Time::endl;
            return hash;
        }();

        const auto value = HashSelection::process(words, hash);
        if (value.has_value()) Time::cout << "Completed: " << *value << Time::endl;
            else Time::cout << "Combinations completed. No matches found." << Time::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed: " << e.what() << std::endl;
    }
    return 0;
}