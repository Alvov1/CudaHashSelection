#include "TimeLogger.h"

namespace HashSelection {
    TimeLogger& HashSelection::TimeLogger::operator<<(const HashSelection::Word& word) {
        return *this << word.data;
    }
}