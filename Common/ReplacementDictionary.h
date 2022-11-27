#ifndef HASHSELECTION_REPLACEMENTDICTIONARY_H
#define HASHSELECTION_REPLACEMENTDICTIONARY_H

#include <functional>
#include <optional>
#include <cstdio>
#include <random>

#include "Word.h"

template <typename Char = char>
class ReplacementDictionary final {
    static constexpr const Word<Char> dictionary[] = {
            { std::is_same<Char, char>::value ? "ÀÁÂÃÄÅÆàáâãäåæĀāĂăĄą" : L"ÀÁÂÃÄÅÆàáâãäåæĀāĂăĄą" },
            { std::is_same<Char, char>::value ? "Þßþ" : L"Þßþ" },
            { std::is_same<Char, char>::value ? "ÇçĆćĈĉĊċČč" : L"ÇçĆćĈĉĊċČč" },
            { std::is_same<Char, char>::value ? "ĎďĐđÐ" : L"" },
            { std::is_same<Char, char>::value ? "ĒēĔĕĖėĘęĚěèéêëÈÉÊË" : L"" },
            { std::is_same<Char, char>::value ? "" : L"" },
            { std::is_same<Char, char>::value ? "ĜðĝĞğĠġĢģ" : L"" },
            { std::is_same<Char, char>::value ? "ĤĥĦħ" : L"" },
            { std::is_same<Char, char>::value ? "ÌÍÎÏìíîïĨĩĪīĬĭĮįİı" : L"" },
            { std::is_same<Char, char>::value ? "ĲĳĴĵ" : L"" },
            { std::is_same<Char, char>::value ? "Ķķĸ" : L"" },
            { std::is_same<Char, char>::value ? "ĹĺĻļĽľĿŀŁł" : L"" },
            { std::is_same<Char, char>::value ? "" : L"" },
            { std::is_same<Char, char>::value ? "ÑñŃńŅņŇňŉŊŋ" : L"" },
            { std::is_same<Char, char>::value ? "òóôõöøÒÓÔÕÖØŌōŎŏŐő" : L"" },
            { std::is_same<Char, char>::value ? "" : L"" },
            { std::is_same<Char, char>::value ? "" : L"" },
            { std::is_same<Char, char>::value ? "ŔŕŖŗŘř" : L"" },
            { std::is_same<Char, char>::value ? "ŚśŜŝŞşŠš" : L"" },
            { std::is_same<Char, char>::value ? "ŢţŤťŦŧ" : L"" },
            { std::is_same<Char, char>::value ? "ÙÚÛÜùúûüŨũŪūŬŭŮůŰűŲų" : L"" },
            { std::is_same<Char, char>::value ? "" : L"" },
            { std::is_same<Char, char>::value ? "Ŵŵ" : L"" },
            { std::is_same<Char, char>::value ? "×" : L"" },
            { std::is_same<Char, char>::value ? "ŶŷŸÝýÿ" : L"" },
            { std::is_same<Char, char>::value ? "ŹźŻżŽž" : L"" },
    };
    static constexpr Word empty = Word<Char>(nullptr);

    static bool nextPermutation(const Word<Char>& candidate, const Word<Char>& pattern,
        Buffer<Char>& buffer, const std::function<bool(const Word<Char>&, const Word<Char>&)>& func) {
        if(buffer.filled() == candidate.size()) return func(buffer.toWord(), pattern);

        const unsigned position = buffer.filled();
        buffer.push(candidate[position]);
        if(nextPermutation(candidate, pattern, buffer, func)) return true;
        buffer.pop();

        const Word<char>& variants = ReplacementDictionary::getVariants(candidate[position]);
        for(unsigned i = 0; i < variants.size(); ++i) {
            buffer.push(variants[i]);
            if(nextPermutation(candidate, pattern, buffer, func)) return true;
            buffer.pop();
        }
        return false;
    }
public:
    static constexpr const Word<Char>& getVariants(Char key) {
        if(key >= L'A' && key <= L'Z')
            return dictionary[key - L'A'];
        if(key >= L'a' && key <= L'z')
            return dictionary[key - L'a'];
        return empty;
    }

    static std::optional<Word<Char>> enumerate(const Word<Char>& candidate, const std::wstring& pattern,
        const std::function<bool(const Word<Char>&, const std::wstring&)>& func) {
//    Buffer buffer(candidate.size());
//    if(nextPermutation(candidate, pattern, buffer, func))
//        return { buffer.toWord() };
        if(func(candidate, pattern)) return { candidate };
        return {};
    }
    static std::basic_string<Char> rearrange(const Word<Char>& word) {
        static std::random_device dev;
        static std::mt19937 rng(dev());
        static std::uniform_int_distribution<std::mt19937::result_type> changing(0, 1);

        auto result = word.to_wstring();

        for(auto& ch : result) {
            const Word<char>& variants = getVariants(ch);
            if(variants.empty() || changing(rng) != 0) continue;

            char replacement = variants[0];
            for (unsigned i = 0; i < variants.size() && changing(rng) != 0; ++i)
                replacement = variants[i];

            ch = replacement;
        }

        return result;
    }
};

#endif //HASHSELECTION_REPLACEMENTDICTIONARY_H
