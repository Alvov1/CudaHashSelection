#ifndef HASHSELECTION_HASHSELECTION_H
#define HASHSELECTION_HASHSELECTION_H

#include <optional>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include "HostHash.h"

#define DEVICE __device__
#define GLOBAL __global__

namespace HashSelection {
    DEVICE bool isVowelDevice(Char sym);

    GLOBAL void foundPermutationsDevice(const Word* forWord, const unsigned char* withHash);

    GLOBAL void foundExtensionsDevice(const Word* data);

    /* Makes all stages all-together on DEVICE. */
    std::optional<Word> runDevice(const std::vector<Word>& words, const HostSHA256& forHash);
}

#endif //HASHSELECTION_HASHSELECTION_H
