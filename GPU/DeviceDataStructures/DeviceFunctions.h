#ifndef HASHSELECTION_DEVICEFUNCTIONS_H
#define HASHSELECTION_DEVICEFUNCTIONS_H

#include "DeviceWord.h"
#include "DeviceVector.h"
#include "DevicePointer.h"
#include "DeviceDictionary.h"
#define DEVICE __device__
#define GLOBAL __global__

namespace DeviceFunctions {
    DEVICE bool hashAndCompare(const DeviceVector& candidate, const DeviceWord& requiredHash) {
        return false;
    }


    DEVICE bool nextPermutation(const DeviceWord& candidate,
        const DeviceWord& requiredHash, DeviceVector& buffer, const char* const* mutations) {
        if(buffer.size() == candidate.size())
            return hashAndCompare(buffer, requiredHash);

        const unsigned position = buffer.size();
        buffer.push_back(candidate[position]);
        if(nextPermutation(candidate, requiredHash, buffer, mutations)) return true;
        buffer.pop_back();

        char replacementKey = '0';
        if(candidate[position] >= 'a' && candidate[position] <= 'z')
            replacementKey = candidate[position] - 'a';
        if(candidate[position] >= 'A' && candidate[position] <= 'Z')
            replacementKey = candidate[position] - 'A';
        if(replacementKey == '0') return false;

        const DeviceWord variants(mutations[replacementKey]);
        for(unsigned i = 0; i < variants.size(); ++i) {
            const char variant = variants[i];
            buffer.push_back(variant);
            if(nextPermutation(candidate, requiredHash, buffer, mutations))
                return true;
            buffer.pop_back();
        }
        return false;
    }

    GLOBAL void process(const char* const* passwords, size_t passSize,
                   const char* const* mutations, size_t mutSize,
                   unsigned wordsPerThread, const char* requiredHash, int* result) {
        const auto threadNumber = blockIdx.x * blockDim.x + threadIdx.x;
        auto passwordPosition = threadNumber * wordsPerThread;
        if(passwordPosition >= passSize) return;

        const auto wordsToCheck = (passwordPosition + wordsPerThread < passSize) ?
                wordsPerThread : passSize - passwordPosition;

        for(unsigned i = 0; i < wordsToCheck; ++passwordPosition, ++i) {
            if(*result != -1) return; /* Founded a solution in another thread. */

            const char* current = passwords[passwordPosition];
//            DeviceSHA256
        }
    }


}

#endif //HASHSELECTION_DEVICEFUNCTIONS_H
