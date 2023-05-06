#include "HashSelection.h"

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

    std::vector<Word> foundExtensionsHost(const Word& forWord) {
        const auto& [pattern, patternSize] = forWord;

        /* Storing in stack (Symbol, Number of repeats in pattern, Number of repeats in current copy). */
        using Stack = std::array<std::tuple<Char, uint8_t, uint8_t>, WordSize>;
        Stack stack {}; unsigned stackPosition {};

        unsigned position = 0;

        /* Skipping first N non-vowel characters inside pattern. */
        for (; !isVowel(pattern[position]) && position < patternSize; ++position)
            stack[stackPosition++] = { pattern[position], 1, 1 };

        std::vector<Word> extensions;

        do {
            if (position < patternSize) {

                /* Count the number of repetition vowels. */
                unsigned vowelsCount = 1;
                for (unsigned i = position + 1; isVowel(pattern[i]) && pattern[i] == pattern[position]; ++vowelsCount, ++i);

                /* Pushing new value in stack */
                stack[stackPosition++] = {
                        pattern[position],
                        vowelsCount,
                        (isVowel(pattern[position]) && vowelsCount == 1) ? 2 : vowelsCount
                };
                position += vowelsCount;

            } else {

                /* Found new pattern. Pushing into buffer. */
                [](Stack& stack, std::vector<Word>& results, unsigned len) {
                    auto& [tData, tSize] = results.emplace_back();
                    for (const auto& [sym, _, reps]: stack)
                        for (unsigned i = 0; i < reps && tSize < WordSize; ++i)
                            tData[tSize++] = sym;
                }(stack, extensions, patternSize);

                /* Popping values from the stack until it's empty or another vowel is found. */
                Char ch = 0; uint8_t reps = 0, repsNow = 0;
                do {
                    std::tie(ch, reps, repsNow) = stack.back();
                    --stackPosition; position -= reps;
                } while (stackPosition != 0 && repsNow < 2);

                if (repsNow-- > 1)
                    stack[stackPosition++] = { ch, reps, repsNow };
                position += reps;

            }
        } while (stackPosition != 0);

        return extensions;
    }

    GLOBAL void foundExtensionsDevice(const Word *data) {
        const unsigned threadNumber = threadIdx.x + blockIdx.x * blockDim.x;
        if(threadNumber > 0) return;

        Word extensions[WordSize]; unsigned foundExtensions = 0;

        [&extensions, &foundExtensions](const Word& word){
            const auto& [pattern, patternSize] = word;

            struct StackElem { Char sym {}; uint8_t reps {}, repsNow {}; };
            StackElem stack[WordSize]; unsigned stackPosition{};

            unsigned position = 0;

            for (; !isVowelDevice(pattern[position]) && position < patternSize; ++position)
                stack[stackPosition++] = {pattern[position], 1, 1};

            do {
                if (position < patternSize) {
                    /* Count the number of repetition vowels. */
                    uint8_t vowelsCount = 1;
                    for (unsigned i = position + 1; isVowelDevice(pattern[i]) && pattern[i] == pattern[position]; ++vowelsCount, ++i);

                    /* Pushing new value in stack */
                    stack[stackPosition++] = {
                            pattern[position],
                            vowelsCount,
                            (isVowelDevice(pattern[position]) && vowelsCount == 1) ? static_cast<uint8_t>(2) : vowelsCount
                    };
                    position += vowelsCount;
                } else {
                    /* Found new pattern. Pushing into buffer. */
                    [&stack, &extensions, &foundExtensions] (unsigned len) {
                        auto& [tData, tSize] = extensions[foundExtensions++];
                        for(const auto& [sym, _, repsNow]: stack)
                            for(unsigned i = 0; i < repsNow && tSize < WordSize; ++i)
                                tData[tSize++] = sym;
                    } (patternSize);

                    Char ch = 0; uint8_t reps = 0, repsNow = 0;
                    do {
                        ch = stack[stackPosition - 1].sym;
                        reps = stack[stackPosition - 1].reps;
                        repsNow = stack[stackPosition - 1].repsNow;
                        --stackPosition; position -= reps;
                    } while (stackPosition != 0 && repsNow < 2);

                    if (repsNow-- > 1)
                        stack[stackPosition++] = { ch, reps, repsNow };
                    position += reps;
                }
            } while (stackPosition != 0);

        } (data[29]);

        for(unsigned i = 0; i < foundExtensions; ++i)
            printf("Found extension: %s\n", extensions[i].data);
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

    std::optional<Word> runDevice(const std::vector<Word> &words, const Closure &onClosure) {
        const thrust::device_vector<HashSelection::Word> deviceWords = words;

        foundExtensionsDevice<<<32, 32>>>(thrust::raw_pointer_cast(deviceWords.data()));
        if(cudaDeviceSynchronize() != cudaSuccess)
            throw std::runtime_error("Failed.");

        return {};
    }
}
