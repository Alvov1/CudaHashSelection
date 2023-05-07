#ifndef HASHSELECTION_HASHSELECTIONDEVICE_H
#define HASHSELECTION_HASHSELECTIONDEVICE_H

#include <optional>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>

#include "Word.h"
#include "TimeLogger.h"

#include "HostHash.h"
#include "DeviceHash.h"

#define DEVICE __device__
#define GLOBAL __global__

namespace HashSelection {
    DEVICE bool isVowelDevice(Char sym);

    GLOBAL void foundPermutationsDevice(const Word* forWord, const unsigned char* withHash);

    GLOBAL void foundExtensionsDevice(const Word* data);

    /* Makes all stages all-together on DEVICE. */
    std::optional<Word> runDevice(const std::vector<Word>& words, const Hash::HostSHA256& forHash);

    GLOBAL void test();
}

#endif //HASHSELECTION_HASHSELECTIONDEVICE_H
