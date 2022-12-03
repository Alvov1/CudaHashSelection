#ifndef HASHSELECTION_DEVICEFUNCTIONS_H
#define HASHSELECTION_DEVICEFUNCTIONS_H

#include "DeviceHash.h"
#include "DeviceWord.h"
#include "DeviceVector.h"
#include "ArrayOnDevice.h"
#include "DictionaryOnDevice.h"

#define DEVICE __device__
#define GLOBAL __global__

namespace DeviceFunctions {
    DEVICE bool fillSolution(const DeviceVector<char>& value, char** place) {
        place = static_cast<char**>(malloc(sizeof(char*)));
        *place = value.getInternalCopy();
        return true;
    }

    DEVICE bool hashAndCompare(const DeviceVector<char>& candidate, const DeviceWord& requiredHash) {
        DeviceSHA256 hash(candidate.get(), candidate.size());
        printf("Calculated hash is %size\n", hash.toVector().get());
    }

    DEVICE bool nextPermutation(const DeviceWord& candidate, const DeviceWord& requiredHash,
                                DeviceVector<char>& buffer, const char* const* mutations, char** resultBuffer) {
        /* Recursion out. */
        if(buffer.size() == candidate.size() && hashAndCompare(buffer, requiredHash))
            return fillSolution(buffer, resultBuffer);

        /* Recursion continue; */
        const unsigned position = buffer.size();
        buffer.push_back(candidate[position]);
        if(nextPermutation(candidate, requiredHash, buffer, mutations, resultBuffer)) return true;
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
            if(nextPermutation(candidate, requiredHash, buffer, mutations, resultBuffer))
                return true;
            buffer.pop_back();
        }
        return false;
    }

    GLOBAL void process(const char* const* passwords, size_t passSize,
                   const char* const* mutations, unsigned wordsPerThread,
                   const char* requiredHash, char* placeForResult) {
        const auto threadNumber = blockIdx.x * blockDim.x + threadIdx.x;
        if(threadNumber > 0) return;

        auto passwordPosition = threadNumber * wordsPerThread;
        if(passwordPosition >= passSize) return;

        DeviceWord current(passwords[passwordPosition]);
        DeviceSHA256 hash(current.get(), current.size());

        printf("Word: %s, hash: %s. Required hash is %size.\n", current.get(), hash.toVector().get(), requiredHash);

//        DeviceVector<char> buffer(current.length(), '\0');
//        if(nextPermutation(current, requiredHash, buffer, mutations, resultBuffer))
//            return; /* Solution found in our thread. */

    }
}

#endif //HASHSELECTION_DEVICEFUNCTIONS_H
