#ifndef MUTATIONSTEST_WORD_H
#define MUTATIONSTEST_WORD_H

#include <iostream>
#include <filesystem>
#include <fstream>
#include <array>
#include <functional>

/* Using ASCII/UTF letters. */
using Char = wchar_t;

namespace Constants {
    static constexpr std::array vowels = [] () {
        if constexpr (std::is_same<Char, char>::value)
            return std::array { 'a', 'e', 'i', 'o', 'u', 'y' };
        else
            return std::array { L'a', L'e', L'i', L'o', L'u', L'y' };
    } ();

    static constexpr std::array replacements = [] () {
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
    } ();
}

namespace HashSelection {
    /* Maximum word length. */
    static constexpr unsigned WordSizeBorder = 32;

    /* Original word content. */
    using PlainWord = std::basic_string<Char>;

    /* Mutated word content. */
    using Mutation = std::array<Char, WordSizeBorder>;

    std::vector<PlainWord> readFileDictionary(const std::filesystem::path& fromLocation);

    std::vector<Mutation> getExpansionsCompressions(const PlainWord& word) {
        static constexpr unsigned DefaultMutationsSize = 3;
        std::vector<Mutation> mutations(DefaultMutationsSize);

        /* Init first cell as initial word value. */
        [] (const PlainWord& word, Mutation& toMutation) {
            for(unsigned i = 0; i < (word.size() < WordSizeBorder ? word.size() : WordSizeBorder); ++i)
                toMutation[i] = word[i];
        } (word, mutations[0]);
        const Mutation& initialValue = mutations[0];

        /* Iterate over initial letters. Add or remove vowel letters. */
        bool previousLetterVowel = false;
        for(unsigned i = 0; i < word.size(); ++i)
            if(std::find(Constants::vowels.begin(), Constants::vowels.end(), initialValue[i]) != Constants::vowels.end()) {

            } else
                previousLetterVowel = false;

    }

    Mutation foundMutation(const Mutation& forMutation, const std::function<bool(const Mutation&)>& comparator);
}


#endif //MUTATIONSTEST_WORD_H
