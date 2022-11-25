#ifndef HASHSELECTION_WORD_H
#define HASHSELECTION_WORD_H

#include <iostream>
#include <cstdio>

class Word final {
    const wchar_t* ptr = nullptr;
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
    : ptr(word != nullptr ? word : nullptr),
      length(word != nullptr ? wcslen(word) : 0) {};
    static Word createEmpty() { return { nullptr }; }

    size_t size() const { return length; }
    bool empty() const { return length == 0; }

    bool operator==(const Word& other) const {
        if(length != other.length) return false;
        return wcscmp(ptr, other.ptr) == 0;
    }
    bool operator==(const wchar_t* other) const {
        return wcscmp(ptr, other) == 0;
    }

    wchar_t operator[](unsigned position) const {
        if(position < length)
            return ptr[position];
        return 0;
    }
    const wchar_t* c_str() const { return ptr; }

    friend std::wostream& operator<<(std::wostream& stream, const Word& data) {
        if(data.ptr == nullptr) return stream;
        return stream << data.ptr;
    }
};

#endif //HASHSELECTION_WORD_H
