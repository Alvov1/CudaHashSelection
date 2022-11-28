#ifndef HASHSELECTION_SCREENWRITER_H
#define HASHSELECTION_SCREENWRITER_H

#include <string>
#include <chrono>
#include <memory>

#ifdef _WIN32
#include <Windows.h>
#endif

#include "Word.h"

namespace Console {
    static inline struct EndLine {} endl;
}

class ScreenWriter {
    virtual void prepareLine() = 0;
protected:
    ScreenWriter& writeChar(char ch) {
        prepareLine();
        printf("%c", ch);
        return *this;
    };
    ScreenWriter& writeChar(wchar_t ch) {
        prepareLine();
        wprintf(L"%lc", ch);
        return *this;
    };
    ScreenWriter& writeString(const char* string) {
        prepareLine();
        printf("%s", string);
        return *this;
    };
    ScreenWriter& writeString(const wchar_t* string) {
        prepareLine();
        wprintf(L"%10ls", string);
        return *this;
    };
public:
    ScreenWriter() {
#ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
#endif
    }

    template <typename PointerBase>
    ScreenWriter& operator<<(const PointerBase* data) {
        return this->writeString(data);
    };

    template <typename Char>
    ScreenWriter& operator<<(const Word<Char>& word) {
        return this->writeString(word.c_str());
    };

    template <typename Char>
    ScreenWriter& operator<<(const std::basic_string<Char>& data) {
        return this->writeString(data.c_str());
    };

    ScreenWriter& operator<<(char ch) {
        return this->writeChar(ch);
    }

    ScreenWriter& operator<<(wchar_t ch) {
        return this->writeChar(ch);
    }

    template <typename Numeric>
    ScreenWriter& operator<<(Numeric value) {
        return writeString(std::to_string(value).c_str());
    };

    virtual ScreenWriter& operator<<(const Console::EndLine&) {
        return this->writeString("\n");
    };
};

class TimedWriter final: public ScreenWriter {
    const std::chrono::system_clock::time_point begin = std::chrono::system_clock::now();
    bool newline = true;

    void prepareLine() override {
        if(!newline) return;
        newline = false;

        using namespace std::chrono;
        wprintf(L"[%lli ms] ", duration_cast<milliseconds>(system_clock::now() - begin).count());
    };
public:
    using ScreenWriter::operator<<;
    TimedWriter& operator<<(const Console::EndLine&) override {
        newline = true;
        printf("\n");
        return *this;
    };
};

class ConsoleWriter final: public ScreenWriter {
    void prepareLine() override {};
};

namespace Console {
    static TimedWriter timer;
    static inline ConsoleWriter out;
};

#endif //HASHSELECTION_SCREENWRITER_H
