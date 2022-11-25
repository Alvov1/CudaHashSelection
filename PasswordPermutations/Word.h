#ifndef HASHSELECTION_WORD_H
#define HASHSELECTION_WORD_H

#include <iostream>
#include <cstdio>

class Word final {
    const wchar_t* word = nullptr;
    size_t length = 0;

    static constexpr size_t wcslen(const wchar_t *s){
        const wchar_t* p= s;
        while (*p)
            p++;
        return p - s;
    }

    static constexpr int wcscmp(const wchar_t *s1, const wchar_t *s2) {
        while (*s1 == *s2++)
            if (*s1++ == '\0')
                return (0);
        /* XXX assumes wchar_t = int */
        return static_cast<int>(*(const unsigned int *)s1 - *(const unsigned int *)--s2);
    }

public:
    constexpr Word(const wchar_t* word)
    : word(word != nullptr ? word : nullptr),
    length(word != nullptr ? wcslen(word) : 0) {};

    static Word createEmpty() {
        return { nullptr };
    }

    size_t size() const {
        return length;
    }
    bool empty() const {
        return length == 0;
    }

    bool operator==(const Word& other) const {
        if(length != other.length) return false;
        return wcscmp(word, other.word) == 0;
    }
    Word operator>>(unsigned shift) const {
        return {word + 1};
    }
    wchar_t operator[](unsigned position) const {
        if(position < length)
            return word[position];
        return 0;
    }

    friend std::wostream& operator<<(std::wostream& stream, const Word& data) {
        if(data.word == nullptr) return stream;
        return stream << data.word;
    }
};

#endif //HASHSELECTION_WORD_H
