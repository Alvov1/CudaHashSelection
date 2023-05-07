#include "Utility.h"

namespace HashSelection {
    Word getRandomModification(const std::vector<Word>& fromWords) {
        static std::mt19937 device(std::random_device {} ());

        /* 1. Get random word from sequence. */
        Word word = [&fromWords] {
            std::uniform_int_distribution<unsigned> dist(0, fromWords.size() - 1);
            return fromWords[dist(device)];
        } ();

        /* 2. Get random word extension. */
        word = [&word] {
            const auto extensions = foundExtensionsHost(word);
            std::uniform_int_distribution<unsigned> dist(0, extensions.size() - 1);
            return extensions[dist(device)];
        } ();

        /* 3. Get random word permutation. */
        [&word] {
            std::uniform_int_distribution<unsigned> dist(0, 1);
            for(unsigned i = 0; i < word.size; ++i)
                for(const auto ch: getVariants(word.data[i]))
                    if(dist(device)) word.data[i] = ch;
        } ();

        return word;
    }

    unsigned long long countComplexity(const std::vector<Word>& words) {
        unsigned long long totalCount = 0;

        for(const auto& word: words) {
            unsigned long long wordCount = 0;

            for(const auto& [data, size]: foundExtensionsHost(word)) {
                unsigned long long extendedWordCount = 1;
                for (unsigned i = 0; i < size; ++i) {
                    const auto variantsSize = getVariants(data[i]).size();
                    extendedWordCount *= (variantsSize > 0 ? variantsSize : 1);
                }
                wordCount += extendedWordCount;
            }
            totalCount += wordCount;
        }

        return totalCount;
    }
}
