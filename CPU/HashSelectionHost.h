#ifndef HASHSELECTION_HASHSELECTIONHOST_H
#define HASHSELECTION_HASHSELECTIONHOST_H

#include <functional>
#include <optional>

#include "HostHash.h"
#include "TimeLogger.h"

namespace HashSelection {
    /* Make all stages all-together on HOST. */
    using Closure = std::function<bool(const Word&)>;
    std::optional<Word> run(const std::vector<Word>& words, const Closure& onClosure);
}


#endif //HASHSELECTION_HASHSELECTIONHOST_H
