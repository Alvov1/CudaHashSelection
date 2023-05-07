#include "HashSelectionDevice.h"

namespace HashSelection {
    GLOBAL void foundPermutationsDevice(const ExtensionList* words, const unsigned char *withHash, Word* resultPlace) {
        const unsigned threadNumber = threadIdx.x + blockIdx.x * blockDim.x;
        if (threadNumber > 127) return;

        const auto &[extensions, foundExtensions] = words[threadNumber];

        struct Stack final {
            struct StackElem final {
                Char sym{};
                short amount{};
            } buffer[WordSize];
            uint8_t position{};
            DEVICE uint8_t push(Char sym, short amount) {
                if (position + 1 < WordSize)
                    buffer[position] = {sym, amount};
                return ++position;
            }
            DEVICE StackElem pop() {
                if (position > 0)
                    return buffer[--position];
                return buffer[0];
            }
            DEVICE Word toWord() const {
                Word result{};
                for (uint8_t i = 0; i < position; ++i)
                    result.data[result.size++] = buffer[i].sym;
                return result;
            }
            DEVICE bool empty() const { return position == 0; }
        } stack;

        for(unsigned i = 0; i < foundExtensions; ++i) {
            const auto& [pattern, patternSize] = extensions[i];

            stack.push(pattern[0], -1);

            while (!stack.empty()) {
                if (stack.position >= patternSize) {
                    const bool found = [&withHash] (const Word &word) {
                        Hash::DeviceSHA256 hash(word.data, word.size * sizeof(Char));
                        for(unsigned i = 0; i < 32; ++i)
                            if(withHash[i] != hash.get()[i]) return false;

                        /* Found coincidence. */
                        printf("Found coincidence for word: %s\n", word.data); return true;
                    } (stack.toWord());
                    if(found) return;

                    Stack::StackElem current {};
                    do {
                        current = stack.pop();
                        const auto& variants = getVariantsDevice(pattern[stack.position]);
                        if(current.amount + 1 < variants.size) break;
                    } while (!stack.empty());

                    const auto& variants = getVariantsDevice(pattern[stack.position]);
                    if(current.amount + 1 < variants.size || !stack.empty())
                        stack.push(variants[current.amount + 1], -1);
                } else
                    stack.push(pattern[stack.position], -1);
            }

            stack.position = 0;
        }

        printf("Thread %d completed.\n", threadNumber);
    }

    GLOBAL void foundExtensionsDevice(const Word* words, ExtensionList *extensionsTotal) {
        const unsigned threadNumber = threadIdx.x + blockIdx.x * blockDim.x;
        if (threadNumber > 127) return;


        [](const Word &forWord) {
            const auto &[pattern, patternSize] = forWord;

            struct Stack final {
                struct StackElem final {
                    Char sym{};
                    uint8_t reps{}, repsNow{};
                } buffer[WordSize]{};
                uint8_t position{};

                DEVICE uint8_t
                push(Char
                sym,
                uint8_t reps, uint8_t
                repsNow) {
                    if (position + 1 < WordSize)
                        buffer[position] = {sym, reps, repsNow};
                    return ++position;
                }
                DEVICE Word

                toWord() const {
                    Word result{};
                    for (uint8_t i = 0; i < position; ++i) {
                        const StackElem &elem = buffer[i];
                        for (uint8_t j = 0; j < elem.repsNow && result.size < WordSize; ++j)
                            result.data[result.size++] = elem.sym;
                    }
                    return result;
                }

                DEVICE StackElem

                pop() {
                    if (position > 0)
                        return buffer[--position];
                    return buffer[0];
                }

                DEVICE bool empty() const { return position == 0; }
            } stack;

            unsigned wordPosition = 0;

            for (; !isVowelDevice(pattern[wordPosition]) && wordPosition < patternSize; ++wordPosition)
                stack.push(pattern[wordPosition], 1, 1);

            do {
                if (wordPosition < patternSize) {
                    /* Count the number of repetition vowels. */
                    uint8_t vowelsCount = 1;
                    for (unsigned i = wordPosition + 1; isVowelDevice(pattern[i]) && pattern[i] == pattern[wordPosition]; ++vowelsCount, ++i);

                    /* Pushing new value in stack */
                    stack.push(
                            pattern[wordPosition],
                            vowelsCount,
                            (isVowelDevice(pattern[wordPosition]) && vowelsCount == 1) ? static_cast<uint8_t>(2)
                                                                                       : vowelsCount
                    );
                    wordPosition += vowelsCount;
                } else {
                    /* Found new word. Pushing into buffer. */
                    currentList.push(stack.toWord());

                    Stack::StackElem current{};
                    do {
                        current = stack.pop();
                        wordPosition -= current.reps;
                    } while (!stack.empty() && current.repsNow < 2);

                    if (current.repsNow-- > 1)
                        stack.push(current.sym, current.reps, current.repsNow);
                    wordPosition += current.reps;
                }
            } while (!stack.empty());

        }(words[threadNumber]);
    }

    std::optional <Word> runDevice(const std::vector <Word> &words, const Hash::HostSHA256 &hash) {
        const thrust::device_vector<HashSelection::Word> deviceExtensions = [] (const std::vector<Word>& words) {
            const thrust::device_vector <HashSelection::Word> deviceWords = words;
            thrust::device_vector <HashSelection::Word> deviceExtensions(words.size() * 8);
            Time::cout << "Dictionary loaded onto device and space for extensions is allocated." << Time::endl;

            foundExtensionsDevice<<<8, 16>>>(
                    thrust::raw_pointer_cast(deviceWords.data()),             /* Words dictionary. */
                    thrust::raw_pointer_cast(deviceExtensions.data())); /* Placeholder for extensions. */
            if (cudaSuccess != cudaDeviceSynchronize())
                throw std::runtime_error("Founding extensions failed.");

            return deviceExtensions;
        } (words);

        Time::cout << "Word extensions found and loaded." << Time::endl;

        const thrust::host_vector<Word> result = [&hash] (const thrust::device_vector<ExtensionList>& deviceExtensions) {
            const thrust::device_vector<unsigned char> deviceHashPattern = [&hash] {
                const auto &data = hash.get();
                return std::vector<unsigned char>(data.begin(), data.end());
            }();
            thrust::device_vector<Word> deviceResult(1);

            foundPermutationsDevice<<<8, 16>>>(
                    thrust::raw_pointer_cast(deviceExtensions.data()),        /* Extensions dictionary. */
                    thrust::raw_pointer_cast(deviceHashPattern.data()),     /* Required digest value. */
                    thrust::raw_pointer_cast(deviceResult.data()));        /* Placeholder for result. */
            if (cudaSuccess != cudaDeviceSynchronize())
                throw std::runtime_error("Founding permutations failed.");

            return deviceResult;
        } (deviceExtensions);

        if(result[0].size > 0) Time::cout << "Completed: " << result[0] << Time::endl;
            else Time::cout << "Failed." << Time::endl;

        return {};
    }
}