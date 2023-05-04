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
}

#endif //HASHSELECTION_WORD_H
