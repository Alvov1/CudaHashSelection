#ifndef MUTATIONSTEST_HASHSELECTION_H
#define MUTATIONSTEST_HASHSELECTION_H

#include <filesystem>
#include <fstream>
#include <array>
#include <functional>
#include <random>

#include "Word.h"

namespace HashSelection {
    std::vector<Word> readFileDictionary(const std::filesystem::path& fromLocation);

    using Closure = std::function<bool(const Word&)>;
    std::optional<Word> foundPermutations(const Word& forWord, const Closure& onClosure);

    std::vector<Word> foundExtensions(const Word& forWord);

    std::optional<Word> process(const std::vector<Word>& words, const Closure& onClosure);

    Word getRandomModification(const std::vector<Word>& fromWords);
}


#endif //MUTATIONSTEST_HASHSELECTION_H
