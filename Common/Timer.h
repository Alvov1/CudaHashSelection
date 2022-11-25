#ifndef HASHSELECTION_TIMER_H
#define HASHSELECTION_TIMER_H

#include <iostream>
#include <chrono>
#include <sstream>

using namespace std::chrono;

class Stream: public std::wostream {
    class StreamBuffer: public std::wstringbuf {
        const system_clock::time_point begin = system_clock::now();
    public:
        StreamBuffer() = default;

        int sync() override {
            const auto now = system_clock::now();
            std::wcout << "[" << duration_cast<milliseconds>(now - begin).count() << " ms] " << str();
            str(L""); std::wcout.flush();
            return 0;
        }

        ~StreamBuffer() override { if (pbase() != pptr()) StreamBuffer::sync(); }
    } buffer;
public:
    explicit Stream() : std::wostream(&buffer) {}
};

namespace Timer {
    Stream out;
}

#endif //HASHSELECTION_TIMER_H
