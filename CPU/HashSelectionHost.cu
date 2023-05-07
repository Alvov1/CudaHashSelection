#include "HashSelectionHost.h"
#include "TimeLogger.h"

namespace HashSelection {
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

        static constexpr auto isVowel = [] (Char sym) {
            if constexpr (std::is_same<Char, char>::value)
                return (sym == 'a' || sym == 'e' || sym == 'i' || sym == 'o' || sym == 'u' || sym == 'y');
            else
                return (sym == L'a' || sym == L'e' || sym == L'i' || sym == L'o' || sym == L'u' || sym == L'y');
        };

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
