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
        word = [&word] {
//            const auto extensions = foundExtensionsHost(word);
//            std::uniform_int_distribution<unsigned> dist(0, extensions.size() - 1);
//            return extensions[dist(device)];
            return word;
        } ();

        /* 3. Get random word permutation. */
        [&word] {
            std::uniform_int_distribution<unsigned> dist(0, 1);
            for(unsigned i = 0; i < word.size; ++i)
                for(const auto ch: getVariantsHost(word.data[i]))
                    if(dist(device)) word.data[i] = ch;
        } ();

        return word;
    }

    unsigned long long countComplexity(const std::vector<Word>& words) {
        unsigned long long totalCount = 0;

        for(const auto& [data, size]: words) {
            unsigned long long wordCount = 0;

            uint8_t position = 0;
            for(uint8_t i = 0; i < size; ++i) {
                uint8_t vowelsCount = 1;
                for (unsigned i = position + 1; isVowelDevice(pattern[i]) && pattern[i] == pattern[position]; ++vowelsCount, ++i);

            }

//            for(const auto& [data, size]: foundExtensionsHost(word)) {
//                unsigned long long extendedWordCount = 1;
//                for (unsigned i = 0; i < size; ++i) {
//                    const auto variantsSize = getVariants(data[i]).size();
//                    extendedWordCount *= (variantsSize > 0 ? variantsSize : 1);
//                }
//                wordCount += extendedWordCount;
//            }
            totalCount += wordCount;
        }

        return totalCount;
    }

    const std::basic_string_view<Char>& getVariantsHost(Char sym) {
        static constexpr std::array variants = [] {
            if constexpr (std::is_same<Char, char>::value)
                return std::array<const std::string_view, 26> {
                        /* A */ "4@^",     /* B */ "86",      /* C */ "[<(",     /* D */ "",        /* E */ "3&",
                        /* F */ "v",       /* G */ "6&9",     /* H */ "#",       /* I */ "1|/\\!",  /* J */ "]}",
                        /* K */ "(<x",     /* L */ "!127|",   /* M */ "",        /* N */ "^",       /* O */ "0",
                        /* P */ "9?",      /* Q */ "20&9",    /* R */ "972",     /* S */ "3$z2",    /* T */ "7+",
                        /* U */ "v",       /* V */ "u",       /* W */ "v",       /* X */ "%",       /* Y */ "j",
                        /* Z */ "27s"
                };
            else
                return std::array<std::wstring_view, 26> {
                        /* A */ L"4@^",     /* B */ L"86",      /* C */ L"[<(",     /* D */ L"",        /* E */
                                L"3&",
                        /* F */ L"v",       /* G */ L"6&9",     /* H */ L"#",       /* I */ L"1|/\\!",  /* J */
                                L"]}",
                        /* K */ L"(<x",     /* L */ L"!127|",   /* M */ L"",        /* N */ L"^",       /* O */
                                L"0",
                        /* P */ L"9?",      /* Q */ L"20&9",    /* R */ L"972",     /* S */ L"3$z2",    /* T */
                                L"7+",
                        /* U */ L"v",       /* V */ L"u",       /* W */ L"v",       /* X */ L"%",       /* Y */
                                L"j",
                        /* Z */ L"27s"
                };
        } ();
        static constexpr std::basic_string_view<Char> empty = [] {
            if constexpr (std::is_same<Char, char>::value)
                return std::string_view { "" };
            else return std::wstring_view { L"" };
        } ();
        static constexpr auto azAZ = [] {
            if constexpr (std::is_same<Char, char>::value)
                return std::tuple { 'a', 'z', 'A', 'Z' };
            else return std::tuple { L'a', L'z', L'A', L'Z' };
        } (); const auto& [a, z, A, Z] = azAZ;

        if(a <= sym && sym <= z)
            return variants[sym - a];
        if(A <= sym && sym <= Z)
            return variants[sym - A];
        return empty;
    }

    DEVICE const MyStringView& getVariantsDevice(Char sym) {
        if constexpr (std::is_same<Char, char>::value) {
            static constexpr DeviceStringView variants[] = {
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
}
