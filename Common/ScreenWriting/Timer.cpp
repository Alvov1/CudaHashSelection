#include "Timer.h"

void TimedWriter::prepareLine() {
    if(!newline) return;
    newline = false;

    using namespace std::chrono;
    wprintf(L"[%lli ms] ", duration_cast<milliseconds>(system_clock::now() - begin).count());
}

TimedWriter &TimedWriter::operator<<(const Console::EndLine &) {
    newline = true;
    printf("\n");
    return *this;
}