#ifndef HASHSELECTION_REPLACEMENTDICTIONARY_H
#define HASHSELECTION_REPLACEMENTDICTIONARY_H

#include <cstdio>
#include <functional>

#include "Word.h"
#include "WordBuffer.h"

class ReplacementDictionary final {
    static const Word dictionary[];
    static constexpr Word empty = Word(nullptr);

    static bool nextPermutation(const Word& candidate, const Word& pattern,
        WordBuffer& buffer, const std::function<bool(const Word&, const Word&)>& func);
public:
    static constexpr const Word& getVariants(wchar_t key);

    static Word enumerate(const Word& candidate, const Word& pattern,
        const std::function<bool(const Word&, const Word&)>& func);
};

#endif //HASHSELECTION_REPLACEMENTDICTIONARY_H
