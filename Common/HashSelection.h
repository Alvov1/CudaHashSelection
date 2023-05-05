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

    /* Get random mutation for random word from the list. */
    Word getRandomModification(const std::vector<Word>& fromWords);

    /* Found word's permutations: azerty -> @s&r7y. Call closure on each. */
    using Closure = std::function<bool(const Word&)>;
    std::optional<Word> foundPermutations(const Word& forWord, const Closure& onClosure);

    /* Prepare word's extensions: home -> { hoome, homee, hoomee }. */
    std::vector<Word> foundExtensions(const Word& forWord);

    /* Make all stages all-together. */
    std::optional<Word> process(const std::vector<Word>& words, const Closure& onClosure);

    /* Count total amount of mutations for all words. */
    unsigned long long countComplexity(const std::vector<Word>& words);
}


#endif //MUTATIONSTEST_HASHSELECTION_H
