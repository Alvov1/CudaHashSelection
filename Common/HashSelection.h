#ifndef HASHSELECTION_HASHSELECTION_H
#define HASHSELECTION_HASHSELECTION_H

#include <filesystem>
#include <functional>
#include <optional>
#include <fstream>
#include <string>
#include <random>
#include <array>

#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>

#include "HostHash.h"

#include "Word.h"

#define DEVICE __device__
#define GLOBAL __global__

namespace HashSelection {
    /* Reads input dictionary into host array. */
    std::vector<Word> readFileDictionary(const std::filesystem::path& fromLocation);

    /* Get random mutation for random word from the list. */
    Word getRandomModification(const std::vector<Word>& fromWords);

    /* Found word's permutations: azerty -> @s&r7y. Call closure on each. */
    using Closure = std::function<bool(const Word&)>;
    std::optional<Word> foundPermutationsHost(const Word& forWord, const Closure &onClosure);
    DEVICE std::optional<Word> foundPermutationsDevice(const Word& forWord, const Word& withHash);

    /* Prepare word's extensions: home -> { hoome, homee, hoomee }. */
    std::vector<Word> foundExtensionsHost(const Word& forWord);
    GLOBAL void foundExtensionsDevice(const Word* data);

    /* Make all stages all-together on HOST. */
    std::optional<Word> runHost(const std::vector<Word>& words, const Closure& onClosure);

    /* Makes all stages all-together on DEVICE. */
    std::optional<Word> runDevice(const std::vector<Word>& words, const HostSHA256& forHash);

    /* Count total amount of mutations for all words. */
    unsigned long long countComplexity(const std::vector<Word>& words);
}


#endif //HASHSELECTION_HASHSELECTION_H
