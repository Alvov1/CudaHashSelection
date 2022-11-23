#include <iostream>

#include "ReplacementDictionary.h"

int main() {
    const Word& left(L"ABBA");
    const Word& right(L"Tail");

    std::function<bool(const WordBuffer&, const Word&)> func = [](const WordBuffer& buffer, const Word& pattern) {
        static unsigned count = 0;
        std::wcout << "[" << count++ << "]: " << buffer << std::endl;
        return buffer.toWord() == pattern;
    };

    const auto word = ReplacementDictionary::enumerate(left, right, func);

    std::wcout << "Found word: " << word << "." << std::endl;
    return 0;
}