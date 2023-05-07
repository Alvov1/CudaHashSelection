#ifndef HASHSELECTION_HASHSELECTIONHOST_H
#define HASHSELECTION_HASHSELECTIONHOST_H

#include <functional>
#include <optional>

#include "HostHash.h"
#include "Word.h"
#include "TimeLogger.h"

namespace HashSelection {
    /* Found word's permutations: azerty -> @s&r7y. Call closure on each. */
    using Closure = std::function<bool(const Word&)>;
    std::optional<Word> foundPermutationsHost(const Word& forWord, const Closure &onClosure);

    /* Prepare word's extensions: home -> { hoome, homee, hoomee }. */
    std::vector<Word> foundExtensionsHost(const Word& forWord);

    /* Make all stages all-together on HOST. */
    std::optional<Word> runHost(const std::vector<Word>& words, const Closure& onClosure);
}


#endif //HASHSELECTION_HASHSELECTIONHOST_H
