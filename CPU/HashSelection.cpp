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

    std::optional<std::string> foundPermutations(const Word& forWord, const std::function<bool(const Word&)>& onClosure){
        static constexpr std::array replacements = []() {
            if constexpr (std::is_same<Char, char>::value)
                return std::array<std::string_view, 26> {
                        /* A */ "4@^",     /* B */ "86",      /* C */ "[<(",     /* D */ "",        /* E */ "3&",
                        /* F */ "v",       /* G */ "6&9",     /* H */ "#",       /* I */ "1|/\\!",  /* J */ "]}",
                        /* K */ "(<x",     /* L */ "!127|",   /* M */ "",        /* N */ "^",       /* O */ "0",
                        /* P */ "9?",      /* Q */ "20&9",    /* R */ "972",     /* S */ "3$z2",    /* T */ "7+",
                        /* U */ "v",       /* V */ "u",       /* W */ "v",       /* X */ "%",       /* Y */ "j",
                        /* Z */ "27s"
                };
            else
                return std::array<std::wstring_view, 26> {
                        /* A */ L"4@^",     /* B */ L"86",      /* C */ L"[<(",     /* D */ L"",        /* E */ L"3&",
                        /* F */ L"v",       /* G */ L"6&9",     /* H */ L"#",       /* I */ L"1|/\\!",  /* J */ L"]}",
                        /* K */ L"(<x",     /* L */ L"!127|",   /* M */ L"",        /* N */ L"^",       /* O */ L"0",
                        /* P */ L"9?",      /* Q */ L"20&9",    /* R */ L"972",     /* S */ L"3$z2",    /* T */ L"7+",
                        /* U */ L"v",       /* V */ L"u",       /* W */ L"v",       /* X */ L"%",       /* Y */ L"j",
                        /* Z */ L"27s"
                };
        }();
    }

    std::vector<Word> foundExtensions(const Word& forWord) {
        static constexpr auto isVowel = [](Char sym) {
            static constexpr std::array vowels = [] {
                if constexpr (std::is_same<Char, char>::value)
                    return std::array {'a', 'e', 'i', 'o', 'u', 'y'};
                else
                    return std::array {L'a', L'e', L'i', L'o', L'u', L'y'};
            }();
            return std::find(vowels.begin(), vowels.end(), sym) != vowels.end();
        };
        const auto [pattern, patternSize] = forWord;

        /* Storing in stack (Symbol, Number of repeats in pattern, Number of repeats in current copy). */
        using Stack = std::vector<std::tuple<char, uint8_t, uint8_t>>;
        Stack stack;
        stack.reserve(forWord.second);

        unsigned position = 0;

        /* Skipping first N non-vowel characters inside pattern. */
        for (; !isVowel(pattern[position]); ++position)
            stack.emplace_back(pattern[position], 1, 1);

        std::vector<Word> result;

        do {
            if (position < patternSize) {

                /* Count the number of repetition vowels. */
                unsigned vowelsCount = 1;
                for (unsigned i = position + 1;
                     isVowel(pattern[i]) && pattern[i] == pattern[position]; ++vowelsCount, ++i);

                /* Pushing new value in stack */
                stack.emplace_back(pattern[position], vowelsCount,
                                   (isVowel(pattern[position]) && vowelsCount == 1) ? 2 : vowelsCount);
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
}
