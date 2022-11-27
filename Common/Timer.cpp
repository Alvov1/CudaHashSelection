#include "Timer.h"

template <>
TimedWriter& TimedWriter::operator<<(const char* data) {
    if(endline) count();
    printf("%s", data);
    return *this;
}

template <>
TimedWriter& TimedWriter::operator<<(const wchar_t* data) {
    if(endline) count();
    wprintf(L"%s", data);
    return *this;
}

template <>
TimedWriter& TimedWriter::operator<<(const Word<char>& word) {
    if(endline) count();
    printf("%s", word.c_str());
    return *this;
}

template <>
TimedWriter& TimedWriter::operator<<(const Word<wchar_t>& word) {
    if(endline) count();
    wprintf(L"%s", word.c_str());
    return *this;
}

TimedWriter& TimedWriter::operator<<(const std::string& data) {
    if(endline) count();
    printf("%s", data.c_str());
    return *this;
}

TimedWriter &TimedWriter::operator<<(const std::wstring& data) {
    if(endline) count();
    wprintf(L"%s", data.c_str());
    return *this;
}

TimedWriter& TimedWriter::operator<<(const Timer::Endliner &) {
    wprintf(L"\n");
    endline = true;
    return *this;
}
