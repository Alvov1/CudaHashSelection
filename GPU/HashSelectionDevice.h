#ifndef HASHSELECTION_HASHSELECTIONDEVICE_H
#define HASHSELECTION_HASHSELECTIONDEVICE_H

#include <vector>
#include <optional>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>

#include "Utility.h"
#include "TimeLogger.h"
#include "HostHash.h"
#include "DeviceHash.h"

#define HOST __host__
#define DEVICE __device__
#define GLOBAL __global__

namespace HashSelection {
    GLOBAL void foundPermutationsDevice(const Word* words, const unsigned* wordsCount, const unsigned char* withHash, Word* resultPlace);

    void runDevice(const std::vector<Word>& words, const Hash::HostSHA256& forHash);
}

#endif //HASHSELECTION_HASHSELECTIONDEVICE_H
