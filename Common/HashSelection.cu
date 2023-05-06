#include "HashSelection.h"
#include "TimeLogger.h"

namespace HashSelection {
    std::vector<Word> readFileDictionary(const std::filesystem::path& fromLocation) {
        if (!std::filesystem::exists(fromLocation))
            throw std::invalid_argument("Dictionary file is not found");
        if (!std::filesystem::is_regular_file(fromLocation))
            throw std::invalid_argument("Bad dictionary file.");

        return [](const std::filesystem::path& location) {
            std::basic_ifstream<Char> input(location);

            unsigned wordsCount = [&input]() {
                std::basic_string<Char> sizeBuffer(10, Char(0));
                input.getline(sizeBuffer.data(), 10);
                return stoi(sizeBuffer);
            }();

            std::vector<Word> values(wordsCount);
            for (auto& [value, size]: values) {
                static constexpr Char space = [] {
                    if constexpr (std::is_same<Char, char>::value)
                        return ' ';
                    else
                        return L' ';
                }();
                input.getline(value, WordSize, space);

                static constexpr auto strlen = [] {
                    if constexpr (std::is_same<Char, char>::value)
                        return std::strlen;
                    else
                        return std::wcslen;
                }();
                size = strlen(value);
            }

            return values;
        }(fromLocation);
    }

    Word getRandomModification(const std::vector<Word>& fromWords) {
        static std::mt19937 device(std::random_device {} ());

        /* 1. Get random word from sequence. */
        Word word = [&fromWords] {
            std::uniform_int_distribution<unsigned> dist(0, fromWords.size() - 1);
            return fromWords[dist(device)];
        } ();

        /* 2. Get random word extension. */
        word = [&word] {
            const auto extensions = foundExtensionsHost(word);
            std::uniform_int_distribution<unsigned> dist(0, extensions.size() - 1);
            return extensions[dist(device)];
        } ();

        /* 3. Get random word permutation. */
        [&word] {
            std::uniform_int_distribution<unsigned> dist(0, 1);
            for(unsigned i = 0; i < word.size; ++i)
                for(const auto ch: getVariants(word.data[i]))
                    if(dist(device)) word.data[i] = ch;
        } ();

        return word;
    }

    unsigned long long countComplexity(const std::vector<Word>& words) {
        unsigned long long totalCount = 0;

        for(const auto& word: words) {
            unsigned long long wordCount = 0;

            for(const auto& [data, size]: foundExtensionsHost(word)) {
                unsigned long long extendedWordCount = 1;
                for (unsigned i = 0; i < size; ++i) {
                    const auto variantsSize = getVariants(data[i]).size();
                    extendedWordCount *= (variantsSize > 0 ? variantsSize : 1);
                }
                wordCount += extendedWordCount;
            }
            totalCount += wordCount;
        }

        return totalCount;
    }

    std::optional<Word> foundPermutationsHost(const Word& forWord, const Closure &onClosure) {
        const auto& [pattern, patternSize] = forWord;

        std::vector<std::pair<Char, short>> stack; stack.reserve(patternSize);
        stack.emplace_back(pattern[0], -1);

        while(!stack.empty()) {
            if(stack.size() >= patternSize) {
                const Word united = [](const std::vector<std::pair<Char, short>>& stack) {
                    Word word {}; auto& [data, size] = word;
                    for(const auto& [sym, _]: stack)
                        data[size++] = sym;
                    return word;
                } (stack);
                if(onClosure(united))
                    return { united };

                unsigned nextPosition = 0;
                do {
                    nextPosition = stack.back().second + 1;
                    stack.pop_back();

                    const auto& variants = getVariants(pattern[stack.size()]);
                    if(nextPosition < variants.size()) break;
                } while (!stack.empty());

                const auto& variants = getVariants(pattern[stack.size()]);
                if(nextPosition < variants.size() || !stack.empty())
                    stack.emplace_back(variants[nextPosition], nextPosition);
            } else
                stack.emplace_back(pattern[stack.size()], -1);
        }

        return {};
    }

    DEVICE std::optional<Word> foundPermutationsDevice(const Word *forWord, const unsigned char *withHash) {
        ;
    }

    std::vector<Word> foundExtensionsHost(const Word& forWord) {
        const auto& [pattern, patternSize] = forWord;

        /* Storing in stack (Symbol, Number of repeats in pattern, Number of repeats in current copy). */
        struct Stack final {
            struct StackElem final {
                Char sym {}; uint8_t reps {}, repsNow {};
            } buffer[WordSize] {};
            uint8_t position {};

            uint8_t push(Char sym, uint8_t reps, uint8_t repsNow) {
                if(position + 1 < WordSize)
                    buffer[position] = { sym, reps, repsNow };
                return ++position;
            }
            Word toWord() const {
                Word result {};
                for(uint8_t i = 0; i < position; ++i) {
                    const StackElem& elem = buffer[i];
                    for (uint8_t j = 0; j < elem.repsNow && result.size < WordSize; ++j)
                        result.data[result.size++] = elem.sym;
                }
                return result;
            }
            StackElem pop() {
                if(position > 0)
                    return buffer[--position];
                return buffer[0];
            }
            bool empty() const { return position == 0; }
        } stack;

        unsigned position = 0;

        /* Skipping first N non-vowel characters inside pattern. */
        for (; !isVowel(pattern[position]) && position < patternSize; ++position)
            stack.push(pattern[position], 1, 1);

        std::vector<Word> extensions;

        do {
            if (position < patternSize) {

                /* Count the number of repetition vowels. */
                uint8_t vowelsCount = 1;
                for (unsigned i = position + 1; isVowel(pattern[i]) && pattern[i] == pattern[position]; ++vowelsCount, ++i);

                /* Pushing new value in stack */
                stack.push(
                        pattern[position],
                        vowelsCount,
                        (isVowel(pattern[position]) && vowelsCount == 1) ? 2 : vowelsCount
                );
                position += vowelsCount;

            } else {

                /* Found new pattern. Pushing into buffer. */
                extensions.push_back(stack.toWord());

                /* Popping values from the stack until it's empty or another vowel is found. */
                Stack::StackElem current {};
                do {
                    current = stack.pop();
                    position -= current.reps;
                } while (!stack.empty() && current.repsNow < 2);

                if (current.repsNow-- > 1)
                    stack.push(current.sym, current.reps, current.repsNow);
                position += current.reps;

            }
        } while (!stack.empty());

        return extensions;
    }

    GLOBAL void foundExtensionsDevice(const Word* data, ExtensionList* extensionsTotal) {
        const unsigned threadNumber = threadIdx.x + blockIdx.x * blockDim.x;
        if(threadNumber > 127) {
            printf("Bad thread number %d\n", threadNumber);
            return;
        }

        ExtensionList& currentList = extensionsTotal[threadNumber];

        [&currentList](const Word& forWord) {
            const auto& [pattern, patternSize] = forWord;

            struct Stack final {
                struct StackElem final {
                    Char sym {}; uint8_t reps {}, repsNow {};
                } buffer[WordSize] {};
                uint8_t position {};

                DEVICE uint8_t push(Char sym, uint8_t reps, uint8_t repsNow) {
                    if(position + 1 < WordSize)
                        buffer[position] = { sym, reps, repsNow };
                    return ++position;
                }
                DEVICE Word toWord() const {
                    Word result {};
                    for(uint8_t i = 0; i < position; ++i) {
                        const StackElem& elem = buffer[i];
                        for (uint8_t j = 0; j < elem.repsNow && result.size < WordSize; ++j)
                            result.data[result.size++] = elem.sym;
                    }
                    return result;
                }
                DEVICE StackElem pop() {
                    if(position > 0)
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
                    for (unsigned i = position + 1; isVowelDevice(pattern[i]) && pattern[i] == pattern[position]; ++vowelsCount, ++i);

                    /* Pushing new value in stack */
                    stack.push(
                            pattern[position],
                            vowelsCount,
                            (isVowelDevice(pattern[position]) && vowelsCount == 1) ? static_cast<uint8_t>(2) : vowelsCount
                    );
                    position += vowelsCount;
                } else {
                    /* Found new forWord. Pushing into buffer. */
                    currentList.push(stack.toWord());

                    Stack::StackElem current {};
                    do {
                        current = stack.pop();
                        position -= current.reps;
                    } while (!stack.empty() && current.repsNow < 2);

                    if (current.repsNow-- > 1)
                        stack.push(current.sym, current.reps, current.repsNow);
                    position += current.reps;
                }
            } while (!stack.empty());

        } (data[threadNumber]);
    }

    GLOBAL void show(const ExtensionList* expressions) {
        const unsigned threadNumber = threadIdx.x + blockIdx.x * blockDim.x;
        if(threadNumber > 0) return;

        std::size_t count = 0;
        for(unsigned i = 0; i < 128; ++i) {
            const ExtensionList& currentList = expressions[i];
            count += currentList.foundExtensions;
        }
        printf("Extensions: %d\n", count);
    }

    std::optional<Word> runDevice(const std::vector<Word> &words, const HostSHA256& hash) {
        /* Copy data dictionary and required hash from host to device. */
        const thrust::device_vector<HashSelection::Word> deviceWords = words;
        thrust::device_vector<HashSelection::ExtensionList> deviceExtensions(words.size());
        Time::cout << "Device memory allocated." << Time::endl;

        /* Start process with global function. */
        foundExtensionsDevice<<<8, 16>>>(
                thrust::raw_pointer_cast(deviceWords.data()),
                thrust::raw_pointer_cast(deviceExtensions.data()));
        if(cudaDeviceSynchronize() != cudaSuccess)
            throw std::runtime_error("Failed.");

        show<<<4, 4>>>(thrust::raw_pointer_cast(deviceExtensions.data()));
        if(cudaDeviceSynchronize() != cudaSuccess)
            throw std::runtime_error("Failed.");

//        const thrust::device_vector<unsigned char> deviceHash =
//                std::vector<unsigned char>(hash.get().begin(), hash.get().end());
//        const thrust::device_ptr<Word> resultPlace = thrust::device_malloc<Word>(1);

        return {};
    }

    std::optional<Word> runHost(const std::vector<Word>& words, const Closure& onClosure) {
        for(const auto& word: words) {
            const auto extendedWords = foundExtensionsHost(word);
            for (const auto& extendedWord: extendedWords) {
                auto option2 = foundPermutationsHost(extendedWord, onClosure);
                if (option2.has_value()) return option2;
            }
        }
        return {};
    }
}
