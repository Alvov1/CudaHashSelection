#include "Word.h"

std::vector<HashSelection::PlainWord> HashSelection::readFileDictionary(const std::filesystem::path& fromLocation) {
    if (!std::filesystem::exists(fromLocation))
        throw std::invalid_argument("Dictionary file is not found");
    if (!std::filesystem::is_regular_file(fromLocation))
        throw std::invalid_argument("Bad dictionary file.");

    return [](const std::filesystem::path& location) {
        std::basic_ifstream<Char> input(location);
        unsigned wordsCount {}; input >> wordsCount;

        std::vector<PlainWord> values(wordsCount);
        for (auto& value: values) input >> value;

        return values;
    }(fromLocation);
}
