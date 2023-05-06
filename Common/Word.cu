#include "Word.h"

namespace HashSelection {
    const VariantsArray& getVariants(Char sym) {
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

    bool isVowel(Char sym) {
        static constexpr std::array vowels = [] {
            if constexpr (std::is_same<Char, char>::value)
                return std::array {'a', 'e', 'i', 'o', 'u', 'y'};
            else
                return std::array {L'a', L'e', L'i', L'o', L'u', L'y'};
        }();
        return std::find(vowels.begin(), vowels.end(), sym) != vowels.end();
    }

    __device__ bool isVowelDevice(Char sym) {
        if constexpr (std::is_same<Char, char>::value)
            return (sym == 'a' || sym == 'e' || sym == 'i' || sym == 'o' || sym == 'u' || sym == 'y');
        else
            return (sym == L'a' || sym == L'e' || sym == L'i' || sym == L'o' || sym == L'u' || sym == L'y');
    }
}
