#include <filesystem>

#include "HostHash.h"
#include "TimeLogger.h"
#include "HashSelection.h"

int main() {
    const std::filesystem::path dictionaryLocation("../../Dictionaries/100.txt");
    const auto words = HashSelection::readFileDictionary(dictionaryLocation);
    Time::cout << "Loaded dictionary" << Time::endl;


    const HashSelection::Word mutation = HashSelection::getRandomModification(words);
    HashSelection::Closure closure = [&mutation] (const HashSelection::Word& forWord) {
        static const HostSHA256 hash { mutation.first.data(), mutation.second };

        const auto& [data, size] = forWord;
        const HostSHA256 another(data.data(), size);

//        return std::memcmp(hash.get().data(), another.get().data(), 32);
        return false;
    };
    Time::cout << "Chosen word: " << mutation << Time::endl;


    HashSelection::process(words, closure);
    Time::cout << "Completed." << Time::endl;

    return 0;
}