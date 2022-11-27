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

    static constexpr Word empty = Word<Char>(nullptr);

    static bool nextPermutation(const Word<Char>& candidate, const std::basic_string<Char>& pattern,
        std::basic_string<Char>& buffer, const Comparator& func);
public:
    static constexpr const Word<Char>& getVariants(Char key);

    static std::optional<std::basic_string<Char>> enumerate(const Word<Char>& candidate, const std::basic_string<Char>& pattern, const Comparator & func);

    static std::basic_string<Char> rearrange(const Word<Char>& word);
};

template<typename Char>
bool ReplacementDictionary<Char>::nextPermutation(const Word<Char> &candidate, const std::basic_string<Char> &pattern,
        std::basic_string<Char> &buffer, const Comparator& func) {
    if(buffer.size() == candidate.size()) return func(buffer, pattern);

    const unsigned position = buffer.size();
    buffer.push_back(candidate[position]);
    if(nextPermutation(candidate, pattern, buffer, func)) return true;
    buffer.pop_back();

    const auto& variants = ReplacementDictionary::getVariants(candidate[position]);
    for(unsigned i = 0; i < variants.size(); ++i) {
        buffer.push_back(variants[i]);
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
//    std::basic_string<Char> value(candidate.c_str());
//    if(func(value, pattern)) return { value };
//    return {};
}

template<typename Char>
std::basic_string<Char> ReplacementDictionary<Char>::rearrange(const Word<Char>& word) {
    static std::random_device dev;
    static std::mt19937 rng(dev());
    static std::uniform_int_distribution<std::mt19937::result_type> changing(0, 1);

    auto result = word.to_string();

//    for(auto& ch : result) {
//        const Word<char>& variants = getVariants(ch);
//        if(variants.empty() || changing(rng) != 0) continue;
//
//        char replacement = variants[0];
//        for (unsigned i = 0; i < variants.size() && changing(rng) != 0; ++i)
//            replacement = variants[i];
//
//        ch = replacement;
//    }

    return result;
}

template <>
constexpr const Word<char> ReplacementDictionary<char>::variants[] = {
        { "ÀÁÂÃÄÅÆàáâãäåæĀāĂăĄą" },{ "Þßþ" },{ "ÇçĆćĈĉĊċČč" },{ "ĎďĐđÐ" },
        { "ĒēĔĕĖėĘęĚěèéêëÈÉÊË" },{ "" },{ "ĜðĝĞğĠġĢģ" },{ "ĤĥĦħ" },
        { "ÌÍÎÏìíîïĨĩĪīĬĭĮįİı" },{ "ĲĳĴĵ" },{ "Ķķĸ" },{ "ĹĺĻļĽľĿŀŁł" },
        { "" },{ "ÑñŃńŅņŇňŉŊŋ" },{ "òóôõöøÒÓÔÕÖØŌōŎŏŐő" },{ "" },
        { "" },{ "ŔŕŖŗŘř" },{ "ŚśŜŝŞşŠš" },{ "ŢţŤťŦŧ" },
        { "ÙÚÛÜùúûüŨũŪūŬŭŮůŰűŲų" },{ "" },{ "Ŵŵ" },{ "×" },
        { "ŶŷŸÝýÿ" },{ "ŹźŻżŽž" },
};

template <>
constexpr const Word<wchar_t> ReplacementDictionary<wchar_t>::variants[] = {
        { L"ÀÁÂÃÄÅÆàáâãäåæĀāĂăĄą" },{ L"Þßþ" },{ L"ÇçĆćĈĉĊċČč" },{ L"ĎďĐđÐ" },
        { L"ĒēĔĕĖėĘęĚěèéêëÈÉÊË" },{ L"" },{ L"ĜðĝĞğĠġĢģ" },{ L"ĤĥĦħ" },
        { L"ÌÍÎÏìíîïĨĩĪīĬĭĮįİı" },{ L"ĲĳĴĵ" },{ L"Ķķĸ" },{ L"ĹĺĻļĽľĿŀŁł" },
        { L"" },{ L"ÑñŃńŅņŇňŉŊŋ" },{ L"òóôõöøÒÓÔÕÖØŌōŎŏŐő" },{ L"" },
        { L"" },{ L"ŔŕŖŗŘř" },{ L"ŚśŜŝŞşŠš" },{ L"ŢţŤťŦŧ" },
        { L"ÙÚÛÜùúûüŨũŪūŬŭŮůŰűŲų" },{ L"" },{ L"Ŵŵ" },{ L"×" },
        { L"ŶŷŸÝýÿ" },{ L"ŹźŻżŽž" },
};

template<>
constexpr const Word<char>& ReplacementDictionary<char>::getVariants(char key) {
    if(key >= 'A' && key <= 'Z')
        return variants[key - 'A'];
    if(key >= 'a' && key <= 'z')
        return variants[key - 'a'];
    return empty;
}

template<>
constexpr const Word<wchar_t>& ReplacementDictionary<wchar_t>::getVariants(wchar_t key) {
    if(key >= L'A' && key <= L'Z')
        return variants[key - L'A'];
    if(key >= L'a' && key <= L'z')
        return variants[key - L'a'];
    return empty;
}

#endif //HASHSELECTION_REPLACEMENTDICTIONARY_H
