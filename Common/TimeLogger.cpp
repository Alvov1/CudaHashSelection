#include "TimeLogger.h"

namespace HashSelection {
    TimeLogger& HashSelection::TimeLogger::operator<<(const HashSelection::Word& word) {
//        for(unsigned i = 0; i < word.second; ++i)
//            *this << std::hex << static_cast<unsigned>(word.first[i]) << ' ';
//        return *this;
        return *this << word.first.data();
    }
}