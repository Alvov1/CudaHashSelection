#ifndef HASHSELECTION_CONSOLE_H
#define HASHSELECTION_CONSOLE_H

#include "ScreenWriter.h"

class ConsoleWriter final: public ScreenWriter {
    void prepareLine() override {};
};

namespace Console {
    inline ConsoleWriter cout;
}

#endif //HASHSELECTION_CONSOLE_H
