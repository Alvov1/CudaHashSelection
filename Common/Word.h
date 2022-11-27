#ifndef HASHSELECTION_WORD_H
#define HASHSELECTION_WORD_H

#include <iostream>
#include <iomanip>
#include <cstdio>

template <typename Char>
class Word final {
    const Char* ptr = nullptr;
    size_t length = 0;

    static constexpr size_t strlen(const Char *s) {
        const Char* p = s;
        while (*p) p++;
        return p - s;
    };
    static constexpr int strcmp(const Char* s1, const Char* s2) {
        while (*s1 == *s2++)
            if (*s1++ == '\0')
                return (0);
        /* XXX assumes wchar_t = int */
        return static_cast<int>(*(const unsigned*)s1 - *(const unsigned*)--s2);
    }
    constexpr Char* strcpy(Char *s1, const Char* s2) {
        Char *cp = s1;
        while ((*cp++ = *s2++) != 0);
        return s1;
    }
public:
    constexpr Word(const Char *word)
    : ptr(word != nullptr ? word : nullptr),
      length(word != nullptr ? strlen(word) : 0) {}
    static Word createEmpty() { return { nullptr }; }

    size_t size() const { return length; }
    bool empty() const { return length == 0; }

    bool operator==(const Word& other) const {
        return (length == other.length) && (strcmp(ptr, other.ptr) == 0);
    }
    bool operator==(const std::basic_string<Char>& other) const {
        return wcsncmp(ptr, other.c_str(),
                       length < other.size() ? length : other.size()) == 0;
    }

    Char operator[](unsigned position) const {
        return (position < length ? ptr[position] : 0);
    };
    char byteAt(unsigned int position) const {
        return (position < length * (sizeof(wchar_t) / sizeof(char)) ?
                reinterpret_cast<const char*>(ptr)[position] : '\0');
    }
    const Char* c_str() const { return ptr; }
    std::basic_string<Char> to_wstring() const {
        return { ptr };
    }

};

template <typename Char>
class Buffer final {
    Char* buffer = nullptr;
    unsigned fillCount = 0;
    unsigned length = 0;
public:
    explicit Buffer(size_t size)
    : buffer(new Char[size + 1]{}), length(size) {};

    void push(Char letter) {
        if(fillCount >= length) return;
        buffer[fillCount++] = letter;
    }
    void pop() {
        if(fillCount == 0) return;
        fillCount--;
    }

    Char operator[](unsigned index) const {
        if(index < fillCount)
            return buffer[index];
        return 0;
    }

    size_t size() const { return length; }
    size_t filled() const { return fillCount; }
    Word<Char> toWord() const { return { buffer }; }

    ~Buffer() { delete [] buffer; }
};

#endif //HASHSELECTION_WORD_H