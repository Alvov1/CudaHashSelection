#ifndef HASHSELECTION_DICTIONARY_H
#define HASHSELECTION_DICTIONARY_H

#include <iostream>
#include <string>
#include <random>

#include "Word.h"

class Dictionary final {
    static const Word dictionary[];
    Dictionary() = default;
public:
    static size_t size();
    static Word get(unsigned index);
    static Word getRandom();

    Dictionary(const Dictionary& copy) = delete;
    Dictionary& operator=(const Dictionary& assign) = delete;
    Dictionary(Dictionary&& move) = delete;
    Dictionary& operator=(Dictionary&& moveAssign) = delete;
};

#endif //HASHSELECTION_DICTIONARY_H
