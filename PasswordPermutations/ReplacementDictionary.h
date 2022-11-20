#ifndef HASHSELECTION_REPLACEMENTDICTIONARY_H
#define HASHSELECTION_REPLACEMENTDICTIONARY_H

#include <cstdio>

class ReplacementDictionary final {
    inline static constexpr wchar_t dictionary[][7] = {
            {L'∆', L'4', L'@', L'Д', L'а'},
            {L'8', L'6', L'ß', L'в', L'ь'},
            {L'<', L'{', L'[', L'(', L'¢', L'с', L'©'},
            {},
            {L'3', L'£', L'₤', L'€', L'е'},
            {L'7', L'ƒ'},
            {L'9', L'[', L'-', L'6'},
            {L'#', L'4', L'н'},
            {L'1', L'|', L'!', L'9'},
            {L'√', L'9'},
            {},
            {L'|', L'1'},
            {L'м'},
            {L'И', L'и', L'п'},
            {L'0', L'Ø', L'Θ', L'о', L'ө'},
            {L'р'},
            {L'9', L'0'},
            {L'Я', L'®'},
            {L'5', L'$'},
            {L'7', L'+', L'т'},
            {},
            {},
            {L'Ш'},
            {L'×', L'%', L'*', L'Ж'},
            {L'¥', L'Ч', L'ү', L'у'},
            {L'5'},
    };

    static size_t getLength(wchar_t key) {
        if(key < L'A' || key > L'Z') return 0;

        size_t length = 0;
        for(; dictionary[key - L'A'][length] != 0; ++length);

        return length;
    }

public:
    static auto getVariants(wchar_t key) {
        struct LimitedPointer {
            const wchar_t* data = nullptr;
            size_t length = 0;
        };

        if(key >= L'A' && key <= L'Z')
            return LimitedPointer {
            .data = reinterpret_cast<const wchar_t*>(&dictionary[key - L'A']),
            .length = getLength(key)
        };

        return LimitedPointer { nullptr, 0 };
    }
};

#endif //HASHSELECTION_REPLACEMENTDICTIONARY_H
