#ifndef HASHSELECTION_REPLACEMENTDICTIONARY_H
#define HASHSELECTION_REPLACEMENTDICTIONARY_H

#include <functional>
#include <optional>
#include <cstdio>
#include <random>

#include "Word.h"

template <typename Char = char>
class ReplacementDictionary final {
public:
    using Comparator = std::function<bool(const std::basic_string<Char>&, const std::basic_string<Char>&)>;
private:
    static const Word<Char> variants[];

    static inline constexpr Word<Char> empty;

    static bool nextPermutation(const Word<Char>& candidate, const std::basic_string<Char>& pattern,
        std::basic_string<Char>& buffer, const Comparator& func);
public:
    static constexpr const Word<Char>& getVariants(Char key);

    static std::optional<std::basic_string<Char>> enumerate(const Word<Char>& candidate, const std::basic_string<Char>& pattern, const Comparator & func);

    static std::basic_string<Char> rearrange(const Word<Char>& word);

    static void showVariants();
};

template<typename Char>
bool ReplacementDictionary<Char>::nextPermutation(const Word<Char> &candidate, const std::basic_string<Char> &pattern,
        std::basic_string<Char> &buffer, const Comparator& func) {
    /* Recursion out. */
    if(buffer.size() == candidate.size())
        return func(buffer, pattern);

    /* Recursion continue. */
    const unsigned position = buffer.size();
    buffer.push_back(candidate[position]);
    if(nextPermutation(candidate, pattern, buffer, func)) return true;
    buffer.pop_back();

    const Word<Char>& tVariants = ReplacementDictionary::getVariants(candidate[position]);
    for(unsigned i = 0; i < tVariants.size(); ++i) {
        buffer.push_back(tVariants[i]);
        if(nextPermutation(candidate, pattern, buffer, func)) return true;
        buffer.pop_back();
    }
    return false;
}

template<typename Char>
std::optional<std::basic_string<Char>> ReplacementDictionary<Char>::enumerate(const Word<Char> &candidate,
        const std::basic_string<Char> &pattern, const Comparator& func) {
    std::basic_string<Char> buffer; buffer.reserve(candidate.size());
    if(nextPermutation(candidate, pattern, buffer, func))
        return { buffer };
    return {};
}

template<typename Char>
std::basic_string<Char> ReplacementDictionary<Char>::rearrange(const Word<Char>& word) {
    static std::random_device dev;
    static std::mt19937 rng(dev());
    static std::uniform_int_distribution<std::mt19937::result_type> changing(0, 2);

    std::basic_string<Char> result = word.to_string();

    for(auto& ch : result) {
        const Word<Char>& tVariants = getVariants(ch);
        if(tVariants.empty() || changing(rng) != 0) continue;

        Char replacement = tVariants[0];
        for (unsigned i = 0; i < tVariants.size() && changing(rng) != 0; ++i)
            replacement = tVariants[i];

        ch = replacement;
    }

    return result;
}

template <>
constexpr const Word<char> ReplacementDictionary<char>::variants[] = {
        { "4@^да" }, { "8ßв" }, { "[<(с" }, { "д" }, { "3&£е€" },
        { "ƒv" }, { "6&9" }, { "#н" }, { "1|!" }, { "]" },
        { "к<" }, { "!12£7|" }, { "м" }, { "^ทп" }, { "0Øо" },
        { "9р" }, { "20&9" }, { "972®я" }, { "3$z§2" }, { "т7+†" },
        { "vบ" }, { "" }, { "พ" }, { "×" }, { "jу¥" }, { "27s" }
};

template <>
constexpr const Word<wchar_t> ReplacementDictionary<wchar_t>::variants[] = {
        { L"4@^да" }, { L"8ßв" }, { L"[<(с" }, { L"д" }, { L"3&£е€" },
        { L"ƒv" }, { L"6&9" }, { L"#н" }, { L"1|!" }, { L"]" },
        { L"к<" }, { L"!12£7|" }, { L"м" }, { L"^ทп" }, { L"0Øо" },
        { L"9р" }, { L"20&9" }, { L"972®я" }, { L"3$z§2" }, { L"т7+†" },
        { L"vบ" }, { L"" }, { L"พ" }, { L"×" }, { L"jу¥" }, { L"27s" }
};

template<typename Char>
constexpr const Word<Char>& ReplacementDictionary<Char>::getVariants(Char key) {
    if(key >= Char('A') && key <= Char('Z'))
        return variants[key - 'A'];
    if(key >= Char('a') && key <= Char('z'))
        return variants[key - 'a'];
    return empty;
}

template<typename Char>
void ReplacementDictionary<Char>::showVariants() {
    Console::out << "Using such password mutations:" << Console::endl;
    for(Char letter = Char('A'); letter != Char('Z'); ++letter) {
        const auto& tVariants = ReplacementDictionary<Char>::getVariants(letter);
        if(tVariants.size() != 0)
            Console::out << letter << ": " << tVariants << Console::endl;
    }
    Console::out << Console::endl;
}

#endif //HASHSELECTION_REPLACEMENTDICTIONARY_H
