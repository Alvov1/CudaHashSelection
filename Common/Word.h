#ifndef HASHSELECTION_WORD_H
#define HASHSELECTION_WORD_H

#include <iostream>
#include <iomanip>
#include <cstdio>

template <typename Char>
class Word final {
    const Char* ptr = nullptr;
    size_t length = 0;

    static constexpr size_t strlen(const Char *s);
    static constexpr int strcmp(const Char* s1, const Char* s2);
    static constexpr Char* strcpy(Char *s1, const Char* s2);
public:
    constexpr Word(const Char *word);

    size_t size() const { return length; }
    bool empty() const { return length == 0; }
    Char operator[](unsigned position) const;
    const Char* c_str() const { return ptr; }

    bool operator==(const Word& other) const;
    bool operator==(const std::basic_string<Char>& other) const;

    std::basic_string<Char> to_string() const { return { ptr }; }

    friend std::basic_ostream<Char>& operator<<(std::basic_ostream<Char>& stream, const Word& data) { return stream << data.ptr; }

    Word() = default;
    Word(const Word& copy) = default;
    Word& operator=(const Word& assign) = default;
    Word(Word&& move) noexcept = default;
    Word& operator=(Word&& moveAssign) noexcept = default;
};

template<typename Char>
constexpr Word<Char>::Word(const Char *word)
        : ptr(word != nullptr ? word : nullptr),
          length(word != nullptr ? strlen(word) : 0) {}

template<typename Char>
constexpr size_t Word<Char>::strlen(const Char *s) {
    const Char* p = s;
    while (*p) p++;
    return p - s;
}

template<typename Char>
constexpr int Word<Char>::strcmp(const Char *s1, const Char *s2) {
    while (*s1 == *s2++)
        if (*s1++ == '\0')
            return (0);
    /* XXX assumes wchar_t = int */
    return static_cast<int>(*(const unsigned*)s1 - *(const unsigned*)--s2);
}

template<typename Char>
constexpr Char *Word<Char>::strcpy(Char *s1, const Char *s2) {
    Char *cp = s1;
    while ((*cp++ = *s2++) != 0);
    return s1;
}

template<typename Char>
bool Word<Char>::operator==(const Word &other) const {
    return (length == other.length) && (strcmp(ptr, other.ptr) == 0);
}

template<typename Char>
bool Word<Char>::operator==(const std::basic_string<Char> &other) const {
    return wcsncmp(ptr, other.c_str(),
                   length < other.size() ? length : other.size()) == 0;
}

template<typename Char>
Char Word<Char>::operator[](unsigned int position) const {
    return (position < length ? ptr[position] : 0);
}


#endif //HASHSELECTION_WORD_H