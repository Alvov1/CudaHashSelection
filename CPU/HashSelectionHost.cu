#include "HashSelectionHost.h"

namespace HashSelection {
    std::optional<Word> foundExtensionsHost(const Word& forWord, const Closure& onClosure) {
        const auto& [initPattern, initSize] = forWord;

        MyStack<thrust::tuple<Char, uint8_t, uint8_t>> extensionsStack {};
        MyStack<thrust::pair<Char, int8_t>> permutationsStack {};
        unsigned wordPosition = 0;

        /* Skipping first N non-vowel characters inside pattern. */
        for (; !isVowel(initPattern[wordPosition]) && wordPosition < initSize; ++wordPosition)
            extensionsStack.push({initPattern[wordPosition], 1, 1});

        do {
            if (wordPosition < initSize) {
                /* Count the number of repetition vowels. */
                uint8_t vowelsCount = 1;
                for (unsigned i = wordPosition + 1; isVowel(initPattern[i]) && initPattern[i] == initPattern[wordPosition]; ++vowelsCount, ++i);

                /* Pushing new value in stack */
                extensionsStack.push({
                    initPattern[wordPosition],
                    vowelsCount,
                    (isVowel(initPattern[wordPosition]) && vowelsCount == 1) ? uint8_t(2) : vowelsCount
                });
                wordPosition += vowelsCount;

            } else {
                /* Found new pattern. Pushing into buffer. */
                const Word extension = [&extensionsStack] {
                    Word result {}; auto& [data, size] = result;
                    for(unsigned i = 0; i < extensionsStack.size(); ++i)
                        for(unsigned j = 0; j < thrust::get<2>(extensionsStack[i]); ++j)
                            data[size++] = thrust::get<0>(extensionsStack[i]);
                    return result;
                } ();
                const auto value = [&onClosure] (const Word& word, MyStack<thrust::pair<Char, int8_t>>& stack) {
                    const auto& [pattern, patternSize] = word;

                    stack.push({pattern[0], -1});

                    while(!stack.empty()) {
                        if(stack.size() >= patternSize) {
                            const Word permutation = [] (const MyStack<thrust::pair<Char, int8_t>>& stack) {
                                Word word {}; auto& [data, size] = word;
                                for(unsigned i = 0; i < stack.size(); ++i)
                                    data[size++] = stack[i].first;
                                return word;
                            } (stack);
                            if(onClosure(permutation)) return std::optional {permutation };
//                                else Time::cout << "\t\tPermutation " << permutation << " completed." << Time::endl;

                            thrust::pair<Char, int8_t> current {}; auto& [sym, varPos] = current;
                            do {
                                current = stack.pop(); ++varPos;
                                const auto& variants = getVariants(pattern[stack.size()]);
                                if(varPos < variants.size) break;
                            } while (!stack.empty());

                            const auto& variants = getVariants(pattern[stack.size()]);
                            if(varPos < variants.size || !stack.empty())
                                stack.push({variants[varPos], varPos});
                        } else
                            stack.push({pattern[stack.size()], -1});
                    }

                    return std::optional<Word> {};
                } (extension, permutationsStack);
                if(value.has_value()) return value;
//                    else Time::cout << "\tExtension " << extension << " completed." << Time::endl;

                /* Popping values from the stack until it's empty or another vowel is found. */
                thrust::tuple<Char, uint8_t, uint8_t> current {};
                do {
                    current = extensionsStack.pop();
                    wordPosition -= thrust::get<1>(current);
                } while (!extensionsStack.empty() && thrust::get<2>(current) < 2);

                if (thrust::get<2>(current)-- > 1)
                    extensionsStack.push(current);
                wordPosition += thrust::get<1>(current);
            }
        } while (!extensionsStack.empty());

        return std::optional<Word> {};
    }

    std::optional<Word> run(const std::vector<Word>& words, const Closure& onClosure) {
        for(const auto& word: words) {
            const auto value = foundExtensionsHost(word, onClosure);
            if(value.has_value()) return value;
                else Time::cout << "Word " << word << " completed." << Time::endl;
        }

        return std::optional<Word> {};
    }
}
