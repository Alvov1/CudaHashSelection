#ifndef HASHSELECTION_WORD_H
#define HASHSELECTION_WORD_H

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <array>
#include <random>

#define DEVICE __device__
#define GLOBAL __global__

namespace HashSelection {
    /* Using ASCII/UTF letters. */
    using Char = char;

    /* Checking passwords up to 31-character long and storing them as pairs of (Data, Size). */
    static constexpr auto WordSize = 32;
    struct Word final {
        Char data[WordSize] {};
        unsigned size {};
    };

    static constexpr auto ExtensionBound = 64;
    struct ExtensionList final {
        Word list[ExtensionBound] {};
        uint8_t foundExtensions {};
        DEVICE uint8_t push(const Word& word) {
            if(foundExtensions + 1 < ExtensionBound)
                list[foundExtensions] = word;
            return ++foundExtensions;
        }
        DEVICE const Word& top() const {
            if(foundExtensions > 0)
                return list[foundExtensions - 1];
            return list[0];
        }
    };

    /* Reads input dictionary into host array. */
    std::vector<Word> readFileDictionary(const std::filesystem::path& fromLocation);
}

#endif //HASHSELECTION_WORD_H
