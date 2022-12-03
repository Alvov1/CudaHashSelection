#ifndef HASHSELECTION_SCREENWRITER_H
#define HASHSELECTION_SCREENWRITER_H

#include <string>
#include <chrono>
#include <memory>

#ifdef _WIN32
#include <Windows.h>
#endif

namespace Console {
    static inline struct EndLine {} endl;
}

class ScreenWriter {
    virtual void prepareLine() = 0;
protected:
    ScreenWriter& writeChar(char ch);;
    ScreenWriter& writeChar(wchar_t ch);;
    ScreenWriter& writeString(const char* string);;
    ScreenWriter& writeString(const wchar_t* string);;
public:
    ScreenWriter() {
#ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
#endif
    }

    template <typename PointerBase, unsigned size>
    ScreenWriter& operator<<(const PointerBase value[size]) {
        return this->writeString(value);
    }

    template <typename PointerBase>
    ScreenWriter& operator<<(const PointerBase* data) {
        return this->writeString(data);
    };

    template <typename Char>
    ScreenWriter& operator<<(const std::basic_string<Char>& data) {
        return this->writeString(data.c_str());
    };

    template <typename Numeric>
    ScreenWriter& operator<<(Numeric value) {
        return writeString(std::to_string(value).c_str());
    };

    ScreenWriter& operator<<(char ch);

    ScreenWriter& operator<<(wchar_t ch);

    virtual ScreenWriter& operator<<(const Console::EndLine&) = 0;
};

#endif //HASHSELECTION_SCREENWRITER_H
