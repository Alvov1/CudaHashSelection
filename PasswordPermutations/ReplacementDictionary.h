#ifndef HASHSELECTION_REPLACEMENTDICTIONARY_H
#define HASHSELECTION_REPLACEMENTDICTIONARY_H

#include <cstdio>
#include <functional>

#include "Word.h"
#include "WordBuffer.h"

class ReplacementDictionary final {
    inline static constexpr Word dictionary[] = {
            { L"4" }, //{ L"∆4@ДАда" },	/* A */
            { L"b" }, //{ L"86ßВвЬь" },		/* B */
            { L"<{[(¢Сс©" },	/* C */
            { nullptr },		/* D */
            { L"3£₤€Ее" },		/* E */
            { L"7ƒ" },		    /* F */
            { L"9[6" },		/* G */
            { L"#4Нн" },		/* H */
            { L"1|!" },		/* I */
            { L"√" },		    /* J */
            { nullptr },		/* K */
            { L"|1" },		    /* L */
            { L"Мм" },		    /* M */
            { L"Ии" },		    /* N */
            { L"0ØΘОоө" },		/* O */
            { L"Рр" },		    /* P */
            { L"90" },		    /* Q */
            { L"Яя®" },		/* R */
            { L"5$" },		    /* S */
            { L"7+Тт" },		/* T */
            { nullptr },		/* U */
            { nullptr },		/* V */
            { L"ШшЩщ" },		/* W */
            { L"×%*Жж" },		/* X */
            { L"¥ЧчүУу" },		/* Y */
            { L"5" },		    /* Z */
    };
    inline static constexpr Word empty = Word(nullptr);

    static bool nextPermutation(const Word& candidate, const Word& pattern,
        WordBuffer& buffer, const std::function<bool(const WordBuffer&, const Word&)>& func) {
        if(buffer.filled() == candidate.size()) return func(buffer, pattern);

        const unsigned position = buffer.filled();
        buffer.push(candidate[position]);
        if(nextPermutation(candidate, pattern, buffer, func)) return true;
        buffer.pop();

        const Word& variants = ReplacementDictionary::getVariants(candidate[position]);
        for(unsigned i = 0; i < variants.size(); ++i) {
            buffer.push(variants[i]);
            if(nextPermutation(candidate, pattern, buffer, func)) return true;
            buffer.pop();
        }
        return false;
    }

public:
    static constexpr const Word& getVariants(wchar_t key) {
        if(key >= L'A' && key <= L'Z')
            return dictionary[key - L'A'];
        if(key >= L'a' && key <= L'z')
            return dictionary[key - L'a'];
        return empty;
    }

    static Word enumerate(const Word& candidate, const Word& pattern,
        const std::function<bool(const WordBuffer&, const Word&)>& func) {
        WordBuffer buffer(candidate.size());
        if(nextPermutation(candidate, pattern, buffer, func))
            return buffer.toWord();
        return Word::createEmpty();
    }
};

#endif //HASHSELECTION_REPLACEMENTDICTIONARY_H
