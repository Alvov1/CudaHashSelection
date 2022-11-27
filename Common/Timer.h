#ifndef HASHSELECTION_TIMER_H
#define HASHSELECTION_TIMER_H

#include <chrono>
//#include <Windows.h>

#include "Word.h"

namespace Timer {
    struct Endliner final { Endliner() = default; };
    inline Endliner endl;
}
class TimedWriter final {
    const std::chrono::system_clock::time_point begin = std::chrono::system_clock::now();
    bool endline = true;

    void count() {
        using namespace std::chrono;
        const auto now = system_clock::now();
        wprintf(L"[%lli ms] ", duration_cast<milliseconds>(now - begin).count());
        endline = false;
    }
public:
    TimedWriter() { /*SetConsoleOutputCP(CP_UTF8);*/ }

    template <typename Char>
    TimedWriter& operator<<(const Char* data);

    template <typename Char>
    TimedWriter& operator<<(const Word<Char>& word);

    TimedWriter& operator<<(const std::wstring& data);

    TimedWriter& operator<<(const std::string& data);

    TimedWriter& operator<<(const Timer::Endliner&);
};

namespace Timer {
    inline TimedWriter out;
}

#endif //HASHSELECTION_TIMER_H
