#ifndef HASHSELECTION_DEVICEWORD_H
#define HASHSELECTION_DEVICEWORD_H

#include "DeviceVector.h"
#define DEVICE __device__

/* Info: Simply constant char pointer with its length
 * and compare operations in one place. */

class DeviceWord final {
    const char* word = nullptr;
    size_t length = 0;

    DEVICE static int strcmp(const char* s1, const char* s2) {
        for ( ; *s1 == *s2; s1++, s2++)
            if (*s1 == '\0')
                return 0;
        return ((*(unsigned char *)s1 < *(unsigned char *)s2) ? -1 : +1);
    }
    DEVICE static size_t strlen(const char *str) {
        const char *s;
        for (s = str; *s; ++s);
        return (s - str);
    }
public:
    DEVICE DeviceWord(const char* word)
    : word(word), length(word != nullptr ? strlen(word) : 0) {};

    DEVICE size_t size() const { return length; }
    DEVICE bool operator==(const DeviceWord& other) const { return strcmp(word, other.word) == 0; }
    template <typename T>
    DEVICE bool operator==(const DeviceVector<T>& other) const { return strcmp(word, static_cast<char*>(other.buffer)) == 0; }
    DEVICE char operator[](unsigned index) const { if(index < length) return word[index]; return '\0'; }
    DEVICE const char* get() const { return word; }

    DEVICE DeviceWord(const DeviceWord& other) = default;
    DEVICE DeviceWord& operator=(const DeviceWord& assign) = default;
    DEVICE DeviceWord(DeviceWord&& move) noexcept = default;
    DEVICE DeviceWord& operator=(DeviceWord&& moveAssign) noexcept = default;
};

#endif //HASHSELECTION_DEVICEWORD_H
