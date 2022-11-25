#ifndef HASHSELECTION_WORDBUFFER_H
#define HASHSELECTION_WORDBUFFER_H

#include <cstdio>
#include "Word.h"

class WordBuffer final {
    wchar_t* buffer = nullptr;
    unsigned count = 0;
    unsigned length = 0;
public:
    explicit WordBuffer(size_t size)
    : buffer(new wchar_t[size + 1]{}), length(size) {};

    void push(wchar_t letter) {
        if(count >= length) return;
        buffer[count++] = letter;
    }
    void pop() {
        if(count == 0) return;
        count--;
    }

    wchar_t operator[](unsigned index) const {
        if(index < count)
            return buffer[index];
        return 0;
    }

    size_t size() const { return length; }
    size_t filled() const { return count; }
    Word toWord() const { return { buffer }; }

    friend std::wostream& operator<<(std::wostream& stream, const WordBuffer& data) {
        return stream << data.buffer;
    }

    ~WordBuffer() { delete [] buffer; }
};

#endif //HASHSELECTION_WORDBUFFER_H
