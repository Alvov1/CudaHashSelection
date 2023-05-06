#include "HashSelection.h"

namespace HashSelection {
    DEVICE bool isVowelDevice(Char sym) {
        if constexpr (std::is_same<Char, char>::value)
        return (sym == 'a' || sym == 'e' || sym == 'i' || sym == 'o' || sym == 'u' || sym == 'y');
        else
        return (sym == L'a' || sym == L'e' || sym == L'i' || sym == L'o' || sym == L'u' || sym == L'y');
    }

    GLOBAL void foundPermutationsDevice(const Word *forWord, const unsigned char *withHash) {
        const unsigned threadNumber = threadIdx.x + blockIdx.x * blockDim.x;
        if (threadNumber > 127) return;

        const auto &[pattern, patternSize] = forWord[threadNumber];

        struct Stack final {
            struct StackElem final {
                Char sym{};
                short amount{};
            } buffer[WordSize];
            uint8_t position{};

            DEVICE uint8_t
            push(Char
                 sym,
                 short amount
            ) {
                if (position + 1 < WordSize)
                    buffer[position] = {sym, amount};
                return ++position;
            }

            DEVICE StackElem

            pop() {
                if (position > 0)
                    return buffer[--position];
                return buffer[0];
            }

            DEVICE bool empty() const { return position == 0; }

            DEVICE Word

            toWord() const {
                Word result{};
                for (uint8_t i = 0; i < position; ++i)
                    result.data[result.size++] = buffer[i].sym;
                return result;
            }
        } stack;
        stack.push(pattern[0], -1);

        while (!stack.empty()) {
            if (stack.position >= patternSize) {
                [&withHash] (const Word& word) {

                } (stack.toWord());
            }
        }
    }

    GLOBAL void foundExtensionsDevice(const Word *data, ExtensionList *extensionsTotal) {
        const unsigned threadNumber = threadIdx.x + blockIdx.x * blockDim.x;
        if (threadNumber > 127) return;

        ExtensionList &currentList = extensionsTotal[threadNumber];

        [&currentList](const Word &forWord) {
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

            unsigned position = 0;

            for (; !isVowelDevice(pattern[position]) && position < patternSize; ++position)
                stack.push(pattern[position], 1, 1);

            do {
                if (position < patternSize) {
                    /* Count the number of repetition vowels. */
                    uint8_t vowelsCount = 1;
                    for (unsigned i = position + 1;
                         isVowelDevice(pattern[i]) && pattern[i] == pattern[position]; ++vowelsCount, ++i);

                    /* Pushing new value in stack */
                    stack.push(
                            pattern[position],
                            vowelsCount,
                            (isVowelDevice(pattern[position]) && vowelsCount == 1) ? static_cast<uint8_t>(2)
                                                                                   : vowelsCount
                    );
                    position += vowelsCount;
                } else {
                    /* Found new forWord. Pushing into buffer. */
                    currentList.push(stack.toWord());

                    Stack::StackElem current{};
                    do {
                        current = stack.pop();
                        position -= current.reps;
                    } while (!stack.empty() && current.repsNow < 2);

                    if (current.repsNow-- > 1)
                        stack.push(current.sym, current.reps, current.repsNow);
                    position += current.reps;
                }
            } while (!stack.empty());

        }(data[threadNumber]);
    }

    std::optional <Word> runDevice(const std::vector <Word> &words, const HostSHA256 &hash) {
        /* Copy data dictionary and required hash from host to device. */
        const thrust::device_vector <HashSelection::Word> deviceWords = words;
        thrust::device_vector <HashSelection::ExtensionList> deviceExtensions(words.size());
        Time::cout << "Device memory allocated." << Time::endl;

        /* Start process with global function. */
        foundExtensionsDevice<<<8, 16>>>(
                thrust::raw_pointer_cast(deviceWords.data()),
                        thrust::raw_pointer_cast(deviceExtensions.data()));
        if (cudaDeviceSynchronize() != cudaSuccess)
            throw std::runtime_error("Failed.");

//        const thrust::device_vector<unsigned char> deviceHash =
//                std::vector<unsigned char>(hash.get().begin(), hash.get().end());
//        const thrust::device_ptr<Word> resultPlace = thrust::device_malloc<Word>(1);

        return {};
    }
}