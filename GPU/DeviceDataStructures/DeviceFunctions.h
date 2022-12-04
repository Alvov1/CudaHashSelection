#ifndef HASHSELECTION_DEVICEFUNCTIONS_H
#define HASHSELECTION_DEVICEFUNCTIONS_H

#include "DeviceHash.h"
#include "DeviceWord.h"
#include "DeviceStack.h"
#include "DeviceVector.h"
#include "ArrayOnDevice.h"
#include "DictionaryOnDevice.h"

#define DEVICE __device__
#define GLOBAL __global__

namespace DeviceFunctions {
    DEVICE void fillSolution(const char* const data, size_t size, char* buffer) {
        memcpy(buffer, data, sizeof(char) * size);
    }

    template <typename T>
    DEVICE bool hashAndCompare(const T& candidate, const DeviceWord& requiredHash) {
//        static unsigned count = 0;
//        printf("%u. Check candidate: %s. ", count++, candidate.get());
//        for(unsigned i = 0; i < candidate.size(); ++i)
//            printf("%u ", static_cast<int>(candidate[i]));
//        printf("\n");

        DeviceSHA256 hash(candidate.get(), candidate.size());
        const auto& value = hash.toVector();
        return requiredHash == value;
    }

    struct pair { char ch; int position; };

    DEVICE DeviceVector<char> stackToVector(const DeviceStack<pair>& stack) {
        DeviceVector<char> result(stack.size(), '\0');
        for(unsigned i = 0; i < stack.size(); ++i)
            result[i] = stack[i].ch;
        return result;
    }

    DEVICE const char* const getVariants(const char* const* variants, char key) {
        if(key >= 'a' && key <= 'z')
            return variants[key - 'a'];
        if(key >= 'A' && key <= 'Z')
            return variants[key - 'A'];
        return nullptr;
    }

    DEVICE void backtracking(const DeviceWord& candidate, const DeviceWord& requiredHash,
                             const char* const* mutations, char* const placeForResult) {
        DeviceStack<pair> stack(candidate.size() + 1);
        stack.push({candidate[0], -1});

        while(!stack.empty()) {
            if(stack.size() >= candidate.size()) {
                const auto string = stackToVector(stack);
                if(hashAndCompare(string, requiredHash))
                    return fillSolution(string.get(), string.size(), placeForResult);

                unsigned nextPosition = 0;
                do {
                    nextPosition = stack.top().position + 1;
                    stack.pop();

                    DeviceWord variants(getVariants(mutations, candidate[stack.size()]));
                    if(nextPosition < variants.size()) break;
                } while (!stack.empty());

                DeviceWord variants(getVariants(mutations, candidate[stack.size()]));
                if(nextPosition < variants.size() || !stack.empty())
                    stack.push({variants[nextPosition], static_cast<int>(nextPosition)});
            } else
                stack.push({candidate[stack.size()], -1});
        }
    }

    GLOBAL void process(const char* const* passwords, size_t passSize,
                   const char* const* mutations, unsigned wordsPerThread,
                   const char* requiredHash, char* placeForResult) {
        const auto threadNumber = blockIdx.x * blockDim.x + threadIdx.x;
        if(threadNumber >= passSize) return;
//        if(threadNumber != 14) return;

        auto passwordPosition = threadNumber * wordsPerThread;
        if(passwordPosition >= passSize) return;

        DeviceWord current(passwords[passwordPosition]);
        DeviceWord required(requiredHash);

//        printf("Candidate: %s\n", current.get());
        backtracking(current, required, mutations, placeForResult);
    }
}

#endif //HASHSELECTION_DEVICEFUNCTIONS_H
