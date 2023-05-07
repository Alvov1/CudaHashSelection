#ifndef HASHSELECTION_HASHSELECTIONDEVICE_H
#define HASHSELECTION_HASHSELECTIONDEVICE_H

#include <vector>
#include <optional>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>

#include "Word.h"
#include "Utility.h"
#include "TimeLogger.h"
#include "HostHash.h"
#include "DeviceHash.h"

#define DEVICE __device__
#define GLOBAL __global__

namespace HashSelection {
    GLOBAL void foundPermutationsDevice(const ExtensionList* words, const unsigned char *withHash, Word* resultPlace);

    GLOBAL void foundExtensionsDevice(const Word* data);

    std::optional<Word> runDevice(const std::vector<Word>& words, const Hash::HostSHA256& forHash);
}

#endif //HASHSELECTION_HASHSELECTIONDEVICE_H
