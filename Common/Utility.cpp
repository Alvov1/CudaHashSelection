#include "Utility.h"

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
                    return ' '; else return L' ';
                }();
                input.getline(value, WordSize, space);

                static constexpr auto strlen = [] {
                    if constexpr (std::is_same<Char, char>::value)
                    return std::strlen; else return std::wcslen;
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
        Word extendedWord = [] (const Word& forWord) {
            const auto& [data, size] = forWord;

            Word newWord {}; auto& [nData, nSize] = newWord;
            for(unsigned i = 0; i < size; ++i) {
                uint8_t vowelsCount = 1;
                for (unsigned j = i + 1; isVowel(data[j]) && data[j] == data[i]; ++vowelsCount, ++j);

                std::uniform_int_distribution<uint8_t> dist(1, (vowelsCount == 1 && isVowel(data[i])) ? 2 : vowelsCount);
                for(uint8_t j = 0; j < dist(device); ++j) nData[nSize++] = data[i];

                i += vowelsCount - 1;
            }
            return newWord;
        } (word);

        /* 3. Get random word permutation. */
        [&extendedWord] {
            std::uniform_int_distribution<unsigned> dist(0, 1);
            for(unsigned i = 0; i < extendedWord.second; ++i)
                for(unsigned j = 0; j < getVariants(extendedWord.first[i]).size; ++j)
                    if(dist(device)) extendedWord.first[i] = getVariants(extendedWord.first[i])[j];
        } ();

        return extendedWord;
    }

    unsigned long long countComplexity(const std::vector<Word>& words, bool verbose) {
        unsigned long long totalCount = 0;

        for(const auto& [data, size]: words) {
            unsigned long long wordCount = 1;

            for(uint8_t i = 0; i < size; ++i) {
                uint8_t vowelsCount = 1;
                for(uint8_t j = i + 1; isVowel(data[j]) && data[j] == data[i]; ++vowelsCount, ++j);
                if(vowelsCount == 1 && isVowel(data[i])) ++vowelsCount;

                const auto& variants = getVariants(data[i]);
                for(uint8_t j = 0; j < vowelsCount; ++j)
                    wordCount *= (variants.size > 0) ? variants.size : 1;

                i += vowelsCount - 1;
            }

            if(verbose)
                std::cout << "Word " << data << " - " << wordCount << " combinations." << std::endl;

            totalCount += wordCount;
        }

        return totalCount;
    }

    HOST DEVICE const MyStringView& getVariants(Char sym) {
        if constexpr (std::is_same<Char, char>::value) {
            static constexpr MyStringView variants[] = {
                    /* A */ "4@^",     /* B */ "86",      /* C */ "[<(",     /* D */ "",        /* E */ "3&",
                    /* F */ "v",       /* G */ "6&9",     /* H */ "#",       /* I */ "1|/\\!",  /* J */ "]}",
                    /* K */ "(<x",     /* L */ "!127|",   /* M */ "",        /* N */ "^",       /* O */ "0",
                    /* P */ "9?",      /* Q */ "20&9",    /* R */ "972",     /* S */ "3$z2",    /* T */ "7+",
                    /* U */ "v",       /* V */ "u",       /* W */ "v",       /* X */ "%",       /* Y */ "j",
                    /* Z */ "27s",     /* Other's */ ""
            };
            if('a' <= sym && sym <= 'z')
                return variants[sym - 'a'];
            if('A' <= sym && sym <= 'Z')
                return variants[sym - 'A'];
            return variants[26];
        } else {
//            static const DeviceStringView variants[] = {
//                    /* A */ L"4@^",     /* B */ L"86",      /* C */ L"[<(",     /* D */ L"",        /* E */ L"3&",
//                    /* F */ L"v",       /* G */ L"6&9",     /* H */ L"#",       /* I */ L"1|/\\!",  /* J */ L"]}",
//                    /* K */ L"(<x",     /* L */ L"!127|",   /* M */ L"",        /* N */ L"^",       /* O */ L"0",
//                    /* P */ L"9?",      /* Q */ L"20&9",    /* R */ L"972",     /* S */ L"3$z2",    /* T */ L"7+",
//                    /* U */ L"v",       /* V */ L"u",       /* W */ L"v",       /* X */ L"%",       /* Y */ L"j",
//                    /* Z */ L"27s",     /* Other's */ L""
//            };
//            if(L'a' <= sym && sym <= L'z')
//                return variants[sym - L'a'];
//            if(L'A' <= sym && sym <= L'Z')
//                return variants[sym - L'A'];
//            return variants[26];
        }
    }

    HOST DEVICE bool isVowel(Char sym) {
        if constexpr (std::is_same<Char, char>::value)
            return (sym == 'a' || sym == 'e' || sym == 'i' || sym == 'o' || sym == 'u' || sym == 'y');
        else
            return (sym == L'a' || sym == L'e' || sym == L'i' || sym == L'o' || sym == L'u' || sym == L'y');
    }
}
