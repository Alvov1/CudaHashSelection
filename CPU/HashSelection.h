#ifndef MUTATIONSTEST_HASHSELECTION_H
#define MUTATIONSTEST_HASHSELECTION_H

#include <filesystem>
#include <functional>
#include <fstream>
#include <random>
#include <array>

#include "Word.h"

namespace HashSelection {
    std::vector<Word> readFileDictionary(const std::filesystem::path& fromLocation);

    Word getRandomModification(const std::vector<Word>& fromWords);

    Word makeWord(const std::basic_string<Char>& str);

    using Closure = std::function<bool(const Word&)>;
    std::optional<Word> foundPermutations(const Word& forWord, const Closure& onClosure);

    std::vector<Word> foundExtensions(const Word& forWord);

    std::optional<Word> process(const std::vector<Word>& words, const Closure& onClosure);

    unsigned long long countComplexity(const std::vector<Word>& words);
}


#endif //MUTATIONSTEST_HASHSELECTION_H
