#ifndef HASHSELECTION_HASHSELECTION_H_H
#define HASHSELECTION_HASHSELECTION_H_H

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <array>
#include <random>

#ifdef CUDA
#include <thrust/pair.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#define HOST __host__
#define GLOBAL __global__
#define DEVICE __device__
#else
#define thrust std
#define HOST
#define DEVICE
#define GLOBAL
#endif


namespace HashSelection {

    /* Using ASCII/UTF letters. */
    using Char = char;

    /* Checking passwords up to 31-character long and storing them as pairs of (Data, Size). */
    static constexpr auto WordSize = 32;
    using Word = thrust::pair<Char[WordSize], unsigned>;

    /* Reads input dictionary into host array. */
    std::vector<Word> readFileDictionary(const std::filesystem::path& fromLocation);

    /* Get random mutation for random word from the list. */
    Word getRandomModification(const std::vector<Word>& fromWords);

    /* Count total amount of mutations for all words. */
    unsigned long long countComplexity(const std::vector<Word>& words, bool verbose = false);

    /* Check vowels */
    HOST DEVICE bool isVowel(Char sym);

    struct MyStringView final {
        const Char* data {};
        std::size_t size {};
        HOST DEVICE constexpr MyStringView() {}
        HOST DEVICE constexpr MyStringView(const Char* dataPtr): data(dataPtr) {
            for(size = 0; dataPtr[size] != 0; ++size);
        };
        HOST DEVICE constexpr Char operator[](std::size_t index) const {
            if(index < size) return data[index];
            return Char();
        }
    };
    HOST DEVICE const MyStringView& getVariants(Char forSym);

    template <typename StackElem, std::size_t buffSize = WordSize>
    class MyStack final {
        StackElem buffer[buffSize];
        uint8_t position{};
    public:
        HOST DEVICE uint8_t push(const StackElem& elem) {
            if (position + 1 < buffSize)
                buffer[position] = elem;
            return ++position;
        }
        HOST DEVICE StackElem pop() {
            if (position > 0)
                return buffer[--position];
            return buffer[0];
        }

        HOST DEVICE StackElem& operator[] (std::size_t index) {
            if(index < position) return buffer[index];
            return buffer[0];
        }
        HOST DEVICE const StackElem& operator[] (std::size_t index) const {
            if(index < position) return buffer[index];
            return buffer[0];
        }

        HOST DEVICE bool empty() const { return position == 0; }
        HOST DEVICE uint8_t size() const { return position; };
        HOST DEVICE StackElem* get() const { return buffer; };
        HOST DEVICE void clear() { position = 0; }
    };
}

#endif //HASHSELECTION_HASHSELECTION_H_H
