#include <filesystem>

#include "HostHash.h"
#include "TimeLogger.h"
#include "HashSelectionHost.h"
#include "Utility.h"

int main() {
    const std::filesystem::path dictionaryLocation("../../Dictionaries/128.txt");
    const auto words = HashSelection::readFileDictionary(dictionaryLocation);
    Time::cout << "Loaded dictionary. Complexity: " << HashSelection::countComplexity(words) << Time::endl;

    const HashSelection::Word mutation = HashSelection::getRandomModification(words);
    HashSelection::Closure closure = [&mutation] (const HashSelection::Word& forWord) {
        static const HostSHA256 hash { mutation.data, mutation.size * sizeof(HashSelection::Char) };
        const HostSHA256 another(forWord.data, forWord.size * sizeof(HashSelection::Char));

        return std::memcmp(hash.get().data(), another.get().data(), 32) == 0;
    };
    Time::cout << "Chosen word: " << mutation << Time::endl;

    const auto value = HashSelection::runHost(words, closure);
    if(value.has_value()) Time::cout << "Completed: " << *value << Time::endl;
        else Time::cout << "Not found." << Time::endl;
    return 0;
}