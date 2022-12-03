#include "ScreenWriter.h"

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

ScreenWriter &ScreenWriter::writeString(const char *string) {
    prepareLine();
    printf("%s", string);
    return *this;
}

ScreenWriter &ScreenWriter::writeString(const wchar_t *string) {
    prepareLine();
    wprintf(L"%10ls", string);
    return *this;
}

ScreenWriter &ScreenWriter::operator<<(char ch) {
    return this->writeChar(ch);
}

ScreenWriter &ScreenWriter::operator<<(wchar_t ch) {
    return this->writeChar(ch);
}
