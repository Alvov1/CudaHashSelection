#ifndef HASHSELECTION_TIMER_H
#define HASHSELECTION_TIMER_H

#include <iostream>
#include <chrono>
#include <sstream>

class Stream: public std::ostream {
    class StreamBuffer: public std::stringbuf {
        const std::chrono::system_clock::time_point begin;
    public:
        StreamBuffer() : begin(std::chrono::system_clock::now()) {}
        ~StreamBuffer() override {
            if (pbase() != pptr()) {
                output();
            }
        }
        int output() {
            const auto now = std::chrono::system_clock::now();
            std::cout << "[" << std::chrono::duration_cast<std::chrono::milliseconds>(now - begin).count() << " ms] " << str();
            str("");
            std::cout.flush();
            return 0;
        }
        inline int sync() override { return output(); }
    };
    StreamBuffer buffer;
public:
    explicit Stream() : std::ostream(&buffer) {}
};

namespace Timer {
    Stream out;
}

#endif //HASHSELECTION_TIMER_H
