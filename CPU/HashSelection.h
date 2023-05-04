#ifndef MUTATIONSTEST_HASHSELECTION_H
#define MUTATIONSTEST_HASHSELECTION_H

#include <iostream>
#include <filesystem>
#include <fstream>
#include <array>
#include <functional>
#include <unordered_map>
#include <map>

namespace HashSelection {
    /* Using ASCII/UTF letters. */
    using Char = wchar_t;

    /* Checking passwords up to 31-character long and storing them as pairs of (Data, Size). */
    static constexpr auto WordSize = 32;
    using Word = std::pair<std::array<Char, WordSize>, unsigned>;

    std::vector<Word> readFileDictionary(const std::filesystem::path& fromLocation);

    using VariantsArray = std::basic_string_view<Char>;
    const VariantsArray& getVariants(Char sym);

    std::optional<Word> foundPermutations(const Word& forWord, const std::function<bool(const Word&)>& onClosure);

    std::vector<Word> foundExtensions(const Word& forWord);
}


#endif //MUTATIONSTEST_HASHSELECTION_H
