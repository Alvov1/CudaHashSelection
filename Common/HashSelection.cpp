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
                input.getline(value.data(), WordSize, space);

                static constexpr auto strlen = [] {
                    if constexpr (std::is_same<Char, char>::value)
                        return std::strlen;
                    else
                        return std::wcslen;
                }();
                size = strlen(value.data());
            }

            return values;
        }(fromLocation);
    }

    std::optional<Word> foundPermutations(const Word& forWord, const std::function<bool(const Word&)>& onClosure){
        const auto& [pattern, patternSize] = forWord;

        std::vector<std::pair<char, short>> stack; stack.reserve(patternSize);
        stack.emplace_back(pattern[0], -1);

        while(!stack.empty()) {
            if(stack.size() >= patternSize) {
                const Word united = [](const std::vector<std::pair<char, short>>& stack) {
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

    std::vector<Word> foundExtensions(const Word& forWord) {
        const auto& [pattern, patternSize] = forWord;

        /* Storing in stack (Symbol, Number of repeats in pattern, Number of repeats in current copy). */
        using Stack = std::vector<std::tuple<char, uint8_t, uint8_t>>;
        Stack stack; stack.reserve(forWord.second);

        unsigned position = 0;

        /* Skipping first N non-vowel characters inside pattern. */
        for (; !isVowel(pattern[position]) && position < patternSize; ++position)
            stack.emplace_back(pattern[position], 1, 1);

        std::vector<Word> result;

        do {
            if (position < patternSize) {

                /* Count the number of repetition vowels. */
                unsigned vowelsCount = 1;
                for (unsigned i = position + 1; isVowel(pattern[i]) && pattern[i] == pattern[position]; ++vowelsCount, ++i);

                /* Pushing new value in stack */
                stack.emplace_back(pattern[position], vowelsCount, (isVowel(pattern[position]) && vowelsCount == 1) ? 2 : vowelsCount);
                position += vowelsCount;

            } else {

                /* Found new pattern. Pushing into buffer. */
                [](Stack& stack, std::vector<Word>& results, unsigned len) {
                    auto& [tData, tSize] = results.emplace_back();
                    for (const auto& [sym, _, reps]: stack)
                        for (unsigned i = 0; i < reps && tSize < WordSize; ++i)
                            tData[tSize++] = sym;
                }(stack, result, patternSize);

                /* Popping values from the stack until it's empty or another vowel is found. */
                char ch = 0;
                uint8_t reps = 0, repsNow = 0;
                do {
                    std::tie(ch, reps, repsNow) = stack.back();
                    stack.pop_back();
                    position -= reps;
                } while (!stack.empty() && repsNow < 2);

                if (repsNow-- > 1)
                    stack.emplace_back(ch, reps, repsNow);
                position += reps;

            }
        } while (!stack.empty());

        return result;
    }

    std::optional<Word> process(const std::vector<Word>& words, const Closure& onClosure) {
        for(const auto& word: words) {
            const auto extendedWords = foundExtensions(word);
            for (const auto& extendedWord: extendedWords) {
                auto option2 = foundPermutations(extendedWord, onClosure);
                if (option2.has_value()) return option2;
            }
        }
        return {};
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
            const auto extensions = foundExtensions(word);
            std::uniform_int_distribution<unsigned> dist(0, extensions.size() - 1);
            return extensions[dist(device)];
        } ();

        /* 3. Get random word permutation. */
        [&word] {
            std::uniform_int_distribution<unsigned> dist(0, 1);
            for(unsigned i = 0; i < word.second; ++i)
                for(const auto ch: getVariants(word.first[i]))
                    if(dist(device)) word.first[i] = ch;
        } ();

        return word;
    }

    unsigned long long countComplexity(const std::vector<Word>& words) {
        unsigned long long totalCount = 0;

        for(const auto& word: words) {
            unsigned long long wordCount = 0;

            for(const auto& [data, size]: foundExtensions(word)) {
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
}
