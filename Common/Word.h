#ifndef HASHSELECTION_WORD_H
#define HASHSELECTION_WORD_H

#include <iostream>
#include <array>

namespace HashSelection {
    /* Using ASCII/UTF letters. */
    using Char = wchar_t;

    /* Checking passwords up to 31-character long and storing them as pairs of (Data, Size). */
    static constexpr auto WordSize = 32;
    using Word = std::pair<std::array<Char, WordSize>, unsigned>;

    /* Storing characters permutation as string-views. */
    using VariantsArray = std::basic_string_view<Char>;
    const VariantsArray& getVariants(Char sym);

    static constexpr auto isVowel = [](Char sym) {
        static constexpr std::array vowels = [] {
            if constexpr (std::is_same<Char, char>::value)
                return std::array {'a', 'e', 'i', 'o', 'u', 'y'};
            else
                return std::array {L'a', L'e', L'i', L'o', L'u', L'y'};
        }();
        return std::find(vowels.begin(), vowels.end(), sym) != vowels.end();
    };
}

#endif //HASHSELECTION_WORD_H
