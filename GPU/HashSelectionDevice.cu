#include "HashSelectionDevice.h"

namespace HashSelection {
    GLOBAL void foundPermutationsDevice(const Word* words, const unsigned* const wordsCount, const unsigned char *withHash, Word* resultPlace) {
        const unsigned threadNumber = threadIdx.x + blockIdx.x * blockDim.x;
        if (threadNumber >= *wordsCount) return;

        const unsigned wordPosition = [] (const Word* words, unsigned wordsCount, unsigned threadNumber) {
            unsigned filledWords = 0, position = 0;

            for(; position < wordsCount && filledWords < threadNumber; ++position)
                if(words[position].second > 0) ++filledWords;

            return position;
        } (words, *wordsCount, threadNumber);
        if(wordPosition == *wordsCount) {
            printf("Bad words sequence - thread %d.\n", threadNumber);
            return;
        }
        const auto& [data, size] = words[wordPosition];

        MyStack<thrust::pair<Char, uint8_t>> stack {};
        stack.push({ data[0], -1 });

        while (!stack.empty()) {
            if (stack.size() >= size) {
                const bool found = [&stack, &withHash] () {
                    /* Assemble stack into word. */
                    Word word {}; auto& [tData, tSize] = word;
                    for(; tSize < stack.size(); ++tSize) tData[tSize] = stack[tSize].first;

                    /* Prepare hash, check the result. */
                    Hash::DeviceSHA256 hash(word.first, word.second * sizeof(Char));
                    for(unsigned i = 0; i < 32; ++i)
                        if(withHash[i] != hash.get()[i]) return false;

                    /* Found coincidence. Fill into placeholder. */
                    printf("Found coincidence for word: %s\n", word.first); return true;
                } ();
                if(found) return;

                thrust::pair<Char, uint8_t> current {}; auto& [sym, varPosition] = current;
                do {
                    current = stack.pop();
                    const auto& variants = getVariants(data[stack.size()]);
                    if(varPosition + 1 < variants.size) break;
                } while (!stack.empty());

                const auto& variants = getVariants(data[stack.size()]);
                if(varPosition + 1 < variants.size || !stack.empty())
                    stack.push({variants[varPosition + 1], -1});
            } else
                stack.push({data[stack.size()], -1});
        }
        printf("Thread %d (word %s) completed.\n", threadNumber, data);
    }

    GLOBAL void foundExtensionsDevice(const Word* words, const unsigned* const wordsCount,
                                      Word* const extensionsTotal, const unsigned* const extensionsCount) {
        const unsigned threadNumber = threadIdx.x + blockIdx.x * blockDim.x;
        if (threadNumber >= *wordsCount) return;

        [](const Word &forWord, Word* const extensionsTotal, const unsigned* const extensionsCount) {
            const auto &[pattern, patternSize] = forWord;
            unsigned wordPosition = 0;

            /* Storing character, repetitions total, repetitions now. */
            MyStack<thrust::tuple<Char, uint8_t, uint8_t>> stack {};

            for (; !isVowel(pattern[wordPosition]) && wordPosition < patternSize; ++wordPosition)
                stack.push({pattern[wordPosition], 1, 1});

            do {
                if (wordPosition < patternSize) {
                    /* Count the number of repetition vowels. */
                    uint8_t vowelsCount = 1;
                    for (unsigned i = wordPosition + 1; isVowel(pattern[i]) && pattern[i] == pattern[wordPosition]; ++vowelsCount, ++i);

                    /* Pushing new value in stack */
                    stack.push({
                        pattern[wordPosition],
                        vowelsCount,
                        (isVowel(pattern[wordPosition]) && vowelsCount == 1) ? static_cast<uint8_t>(2) : vowelsCount});
                    wordPosition += vowelsCount;
                } else {
                    /* Found new word. Pushing into buffer. */
                    [&stack, &extensionsTotal, &extensionsCount] {
                        /* 1. Assemble stack into word. */
                        Word word {}; auto& [data, size] = word;
                        for(uint8_t i = 0; i < stack.size(); ++i)
                            for(uint8_t j = 0; j < thrust::get<2>(stack[i]) && size < WordSize; ++j)
                                data[size++] = thrust::get<0>(stack[i]);

                        /* 2. Found an empty spot for our word. */
                        unsigned extensionPosition = 0;
                        for(; extensionPosition < *extensionsCount; ++extensionPosition) {
                            Word& extension = extensionsTotal[extensionPosition];
                            if(0 == atomicCAS(&extension.second, 0, size)) break;
                        }
                        if(extensionPosition == *extensionsCount) {
                            printf("WARNING! Extension table overflow: sequence '%s' will be lost.\n", word.first);
                            return;
                        }

                        /* 3. Fill our word into founded space. */
                        Word& ourPlace = extensionsTotal[extensionPosition];
                        for(unsigned i = 0; i < word.second; ++i)
                            ourPlace.first[i] = word.first[i];
                    } ();

                    thrust::tuple<Char, uint8_t, uint8_t> current {};
                    do {
                        current = stack.pop();
                        wordPosition -= thrust::get<1>(current);
                    } while (!stack.empty() && thrust::get<2>(current) < 2);

                    if (thrust::get<2>(current)-- > 1)
                        stack.push(current);
                    wordPosition += thrust::get<1>(current);
                }
            } while (!stack.empty());

        } (words[threadNumber], extensionsTotal, extensionsCount);
    }

    std::unique_ptr<Word> runDevice(const std::vector <Word> &words, const Hash::HostSHA256 &hash) {
        const auto extensions = [] (const std::vector<Word>& words) {
            const thrust::device_vector <HashSelection::Word> deviceWords = words;
            const auto deviceWordsCount = thrust::device_malloc<unsigned>(1);
            *deviceWordsCount = words.size();

            /* Allocating space for the number of initial word multiplied by 8. */
            const auto extensionsBorder = words.size() * 8;
            std::pair values = {
                    thrust::device_vector<HashSelection::Word>(extensionsBorder), /* Place for extensions. */
                    thrust::device_malloc<unsigned>(1) /* Size of the allocated buffer. */
            };
            *values.second = extensionsBorder;

            Time::cout << "Dictionary loaded onto device and space for extensions is allocated." << Time::endl;

            foundExtensionsDevice<<<8, 16>>>(
                    thrust::raw_pointer_cast(deviceWords.data()),
                    thrust::raw_pointer_cast(deviceWordsCount.get()),
                    thrust::raw_pointer_cast(values.first.data()),
                    thrust::raw_pointer_cast(values.second.get()));
            if (cudaSuccess != cudaDeviceSynchronize())
                throw std::runtime_error("Founding extensions failed.");

            return values;
        } (words);

        Time::cout << "Word extensions completed." << Time::endl;

        return [&hash, &extensions] {
            const thrust::device_vector<unsigned char> deviceHashPattern =
                std::vector<unsigned> { hash.cbegin(), hash.cend() };
            thrust::device_ptr<Word> deviceResult = thrust::device_malloc<Word>(1);

            foundPermutationsDevice<<<8, 16>>>(
                    thrust::raw_pointer_cast(extensions.first.data()),
                    thrust::raw_pointer_cast(extensions.second.get()),
                    thrust::raw_pointer_cast(deviceHashPattern.data()),
                    thrust::raw_pointer_cast(deviceResult.get()));
            if (cudaSuccess != cudaDeviceSynchronize())
                throw std::runtime_error("Founding permutations failed.");

            std::unique_ptr<Word> hostResult = std::make_unique<Word>();
            thrust::copy(deviceResult, deviceResult + 1, hostResult.get());
            return hostResult;
        } ();
    }
}