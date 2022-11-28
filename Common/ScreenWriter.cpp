#include "ScreenWriter.h"

ScreenWriter& ScreenWriter::writeString(const char* string) {
    prepareLine();
    printf("%s", string);
    return *this;
}

ScreenWriter& ScreenWriter::writeString(const wchar_t* string) {
    prepareLine();
    wprintf(L"%10ls", string);
    return *this;
}

ScreenWriter &ScreenWriter::writeChar(char ch) {
    prepareLine();
    printf("%c", ch);
    return *this;
}

ScreenWriter &ScreenWriter::writeChar(wchar_t ch) {
    prepareLine();
    wprintf(L"%lc", ch);
    return *this;
}

TimedWriter& TimedWriter::operator<<(const Console::EndLine &) {
    newline = true;
    printf("\n");
    return *this;
}

void TimedWriter::prepareLine() {
    if(!newline) return;
    newline = false;

    using namespace std::chrono;
    wprintf(L"[%lli ms] ", duration_cast<milliseconds>(system_clock::now() - begin).count());
}