#ifndef HASHSELECTION_HASHSELECTION_H_H
#define HASHSELECTION_HASHSELECTION_H_H

#include "Word.h"

namespace HashSelection {
    /* Get random mutation for random word from the list. */
    Word getRandomModification(const std::vector<Word>& fromWords);

    /* Count total amount of mutations for all words. */
    unsigned long long countComplexity(const std::vector<Word>& words);

    /* Get permutations for specified character. */
    const std::basic_string_view<Char>& getVariantsHost (Char sym);

    struct DeviceStringView final {
        const Char* data {};
        std::size_t size {};
        DEVICE constexpr DeviceStringView() {}
        DEVICE constexpr DeviceStringView(const Char* dataPtr): data(dataPtr) {
            for(size = 0; dataPtr[size] != 0; ++size);
        };
        DEVICE constexpr Char operator[](std::size_t index) const {
            if(index < size) return data[index];
            return Char();
        }
    };
    DEVICE const DeviceStringView& getVariantsDevice(Char sym);

    /* Check vowels */
    bool isVowelHost(Char sym) {
        if constexpr (std::is_same<Char, char>::value)
            return (sym == 'a' || sym == 'e' || sym == 'i' || sym == 'o' || sym == 'u' || sym == 'y');
        else
            return (sym == L'a' || sym == L'e' || sym == L'i' || sym == L'o' || sym == L'u' || sym == L'y');
    };
    DEVICE bool isVowelDevice(Char sym) {
        if constexpr (std::is_same<Char, char>::value)
            return (sym == 'a' || sym == 'e' || sym == 'i' || sym == 'o' || sym == 'u' || sym == 'y');
        else
            return (sym == L'a' || sym == L'e' || sym == L'i' || sym == L'o' || sym == L'u' || sym == L'y');
    }
}

#endif //HASHSELECTION_HASHSELECTION_H_H
