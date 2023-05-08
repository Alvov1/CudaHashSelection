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

    GLOBAL void foundExtensionsDevice(const Word* words, const unsigned* const wordsCount, const unsigned char* withHash) {
        const unsigned threadNumber = threadIdx.x + blockIdx.x * blockDim.x;
        if (threadNumber > 0) return;

        [] (const Word& forWord, const unsigned char* const withHash) {
            const auto& [pattern, patternSize] = forWord;
            unsigned wordPosition = 0;

            printf("Working with word %s\n", pattern);

            /* Stack for extensions: Storing character, repetitions total, repetitions now. */
            MyStack<thrust::tuple<Char, uint8_t, uint8_t>> extensionStack {};
            /* Stack for permutations: Storing character and variant position. */
            MyStack<thrust::pair<Char, uint8_t>> permutationStack {};

            for (; !isVowel(pattern[wordPosition]) && wordPosition < patternSize; ++wordPosition)
                extensionStack.push({pattern[wordPosition], 1, 1});

            do {
                if (wordPosition < patternSize) {
                    /* Count the number of repetition vowels. */
                    uint8_t vowelsCount = 1;
                    for (unsigned i = wordPosition + 1; isVowel(pattern[i]) && pattern[i] == pattern[wordPosition]; ++vowelsCount, ++i);

                    /* Pushing new value in extensionStack */
                    extensionStack.push({
                        pattern[wordPosition],
                        vowelsCount,
                        (isVowel(pattern[wordPosition]) && vowelsCount == 1) ? static_cast<uint8_t>(2) : vowelsCount});
                    wordPosition += vowelsCount;

                    printf("Pushed into stack: ");
                    for(unsigned i = 0; i < extensionStack.size(); ++i)
                        printf("(%c %d %d), ", pattern[wordPosition], vowelsCount, (isVowel(pattern[wordPosition]) && vowelsCount == 1) ? static_cast<uint8_t>(2) : vowelsCount);
                    printf("\n");
                } else {
                    /* Found new word. Checking permutations. */
                    [&extensionStack, &permutationStack, &withHash] {
                        const Word mutatedWord = [&extensionStack] {
                            Word result {}; auto& [data, size] = result;
                            for (unsigned i = 0; i < extensionStack.size(); ++i)
                                for (unsigned j = 0; j < thrust::get<2>(extensionStack[i]); ++j)
                                    data[size++] = thrust::get<0>(extensionStack[0]);
                            return result;
                        } ();
                        const auto& [data, size] = mutatedWord;
                        printf("Assembled extentend word %s\n", data);

                        permutationStack.push({ data[0], -1 });

                        while (!permutationStack.empty()) {
                            if (permutationStack.size() >= size) {
                                const bool found = [&permutationStack, &withHash] () {
                                    /* Assemble stack into word. */
                                    Word word = [&permutationStack] {
                                        Word result {}; auto& [tData, tSize] = result;
                                        for(unsigned i = 0; i < permutationStack.size(); ++i) {
                                            const auto& [sym, var] = permutationStack[i];
                                            const auto& variants = getVariants(sym);
                                            tData[tSize++] = variants[var];
                                        }
                                        return result;
                                    } ();
                                    printf("Checking %s\n", word.first);

                                    /* Prepare hash, check the result. */
                                    Hash::DeviceSHA256 hash(word.first, word.second * sizeof(Char));
                                    for(unsigned i = 0; i < 32; ++i)
                                        if(withHash[i] != hash.get()[i]) return false;

                                    /* Found coincidence. Fill into placeholder. */
                                    printf("Found coincidence for word: %s\n", word.first); return true;
                                } ();
                                if(found) return;

                                thrust::pair<Char, uint8_t> combination {}; auto& [sym, varPosition] = combination;
                                do {
                                    combination = permutationStack.pop();
                                    const auto& variants = getVariants(data[permutationStack.size()]);
                                    if(varPosition + 1 < variants.size) break;
                                } while (!permutationStack.empty());

                                const auto& variants = getVariants(data[permutationStack.size()]);
                                if(varPosition + 1 < variants.size || !permutationStack.empty())
                                    permutationStack.push({variants[varPosition + 1], -1});
                            } else
                                permutationStack.push({data[permutationStack.size()], -1});
                        }

                        permutationStack.clear();
                    } ();

                    thrust::tuple<Char, uint8_t, uint8_t> current {};
                    do {
                        current = extensionStack.pop();
                        wordPosition -= thrust::get<1>(current);
                    } while (!extensionStack.empty() && thrust::get<2>(current) < 2);

                    printf("Popped from stack until: (%c, %d, %d)\n",
                           thrust::get<0>(current),
                                   thrust::get<1>(current),
                                           thrust::get<2>(current));


                    if (thrust::get<2>(current)-- > 1)
                        extensionStack.push(current);
                    wordPosition += thrust::get<1>(current);
                }
            } while (!extensionStack.empty());

        } (words[threadNumber], withHash);
    }

    void runDevice(const std::vector <Word> &words, const Hash::HostSHA256 &hash) {
        const thrust::device_vector<HashSelection::Word> deviceWords = words;
        const auto deviceWordsCount = thrust::device_malloc<unsigned>(1);
        *deviceWordsCount = words.size();
        const thrust::device_vector<unsigned char> deviceHash = std::vector<unsigned char>{hash.cbegin(), hash.cend()};

        Time::cout << "Dictionary loaded onto device and space for extensions is allocated." << Time::endl;

        foundExtensionsDevice<<<8, 16>>>(
                thrust::raw_pointer_cast(deviceWords.data()),
                thrust::raw_pointer_cast(deviceWordsCount.get()),
                thrust::raw_pointer_cast(deviceHash.data()));
        if (cudaSuccess != cudaDeviceSynchronize())
            throw std::runtime_error("Founding extensions failed.");

        Time::cout << "Word extensions completed." << Time::endl;
    }
}