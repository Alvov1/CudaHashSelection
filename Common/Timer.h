#ifndef HASHSELECTION_TIMER_H
#define HASHSELECTION_TIMER_H

#include <iostream>
#include <chrono>
#include <sstream>

#include "Word.h"

using namespace std::chrono;

namespace Timer {
    struct Endliner final { Endliner() = default; };
    Endliner endl;
}

class Writer final {
    const system_clock::time_point begin = system_clock::now();
    bool endline = true;

    void count() {
        const auto now = system_clock::now();
        wprintf(L"[%I64u ms] ", duration_cast<milliseconds>(now - begin).count());
        endline = false;
    }
public:
    Writer() { SetConsoleOutputCP(CP_UTF8); }

    template <typename BaseType>
    Writer& operator<<(const BaseType* data);

    template <>
    Writer& operator<<(const char* data) {
        if(endline) count();
        printf("%s", data);
        return *this;
    }

    template <>
    Writer& operator<<(const wchar_t* data) {
        if(endline) count();
        wprintf(L"%s", data);
        return *this;
    }

    Writer& operator<<(const std::wstring& data) {
        if(endline) count();
        wprintf(L"%s", data.c_str());
        return *this;
    }

    Writer& operator<<(const std::string& data) {
        if(endline) count();
        printf("%s", data.c_str());
        return *this;
    }

    Writer& operator<<(const Timer::Endliner&) {
        wprintf(L"\n");
        endline = true;
        return *this;
    }

    Writer& operator<<(const Word<wchar_t>& word) {
        if(endline) count();
        wprintf(L"%s", word.c_str());
        return *this;
    }
};

namespace Timer {
    Writer out;
}

#endif //HASHSELECTION_TIMER_H
