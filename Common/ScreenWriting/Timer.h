#ifndef HASHSELECTION_TIMER_H
#define HASHSELECTION_TIMER_H

#include "ScreenWriter.h"

class TimedWriter final: public ScreenWriter {
    const std::chrono::system_clock::time_point begin = std::chrono::system_clock::now();
    bool newline = true;
    void prepareLine() override;;
public:
    using ScreenWriter::operator<<;
    TimedWriter& operator<<(const Console::EndLine&) override;
};

namespace Console {
    inline TimedWriter timer;
}

#endif //HASHSELECTION_TIMER_H