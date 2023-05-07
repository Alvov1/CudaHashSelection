#ifndef HASHSELECTION_HASHSELECTION_H_H
#define HASHSELECTION_HASHSELECTION_H_H

#include "Word.h"
#include "HashSelectionHost.h"

namespace HashSelection {
    /* Get random mutation for random word from the list. */
    Word getRandomModification(const std::vector<Word>& fromWords);

    /* Count total amount of mutations for all words. */
    unsigned long long countComplexity(const std::vector<Word>& words);
}

#endif //HASHSELECTION_HASHSELECTION_H_H
