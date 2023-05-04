#ifndef HASHSELECTION_TIMELOGGER_H
#define HASHSELECTION_TIMELOGGER_H

#include <iostream>
#include <chrono>

#include "Word.h"

namespace Time {
    inline struct Endl {} endl;
}

namespace HashSelection {
    class TimeLogger final {
        using Clock = std::chrono::system_clock;
        Clock::time_point begin = Clock::now();
        bool newLine = true;
    public:
        template<typename T>
        TimeLogger& operator<<(const T& msg);
        TimeLogger& operator<<(const HashSelection::Word& word);
    };

    template <typename T>
    TimeLogger& HashSelection::TimeLogger::operator<<(const T& msg) {
        const auto duration = duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - begin).count();
        if constexpr (std::is_same<Char, char>::value) {
            if(newLine) {
                std::cout << '[' << duration << " ms ] " << msg;
                newLine = false;
            } else std::cout << msg;
        } else {
            if(newLine) {
                std::wcout << L'[' << duration << L" ms ] " << msg;
                newLine = false;
            } else std::wcout << msg;
        }
        return *this;
    }

    template <>
    inline TimeLogger& HashSelection::TimeLogger::operator<<<Time::Endl>(const Time::Endl& msg) {
        if constexpr (std::is_same<Char, char>::value)
            std::cout.write("\n", 1);
        else std::wcout.write(L"\n", 1);
        newLine = true;
        return *this;
    }
}

namespace Time {
    inline HashSelection::TimeLogger cout {};
}

#endif //HASHSELECTION_TIMELOGGER_H
