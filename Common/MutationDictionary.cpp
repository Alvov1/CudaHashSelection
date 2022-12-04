#include "MutationDictionary.h"

const IDictionary::WordArray &MutationDictionary::get() const {
    static std::vector<std::string> words = {
            { "4@^да" }, { "8ßв" }, { "[<(с" }, { "д" }, { "3&£е€" },
            { "ƒv" }, { "6&9" }, { "#н" }, { "1|!" }, { "]" },
            { "к<" }, { "!12£7|" }, { "м" }, { "^ทп" }, { "0Øо" },
            { "9р" }, { "20&9" }, { "972®я" }, { "3$z§2" }, { "т7+†" },
            { "vบ" }, { "" }, { "พ" }, { "×" }, { "jу¥" }, { "27s" }
    };

    return words;
}

std::string MutationDictionary::mutate(const std::string& word) const {
    static std::random_device dev;
    static std::mt19937 rng(dev());
    static std::uniform_int_distribution<std::mt19937::result_type> changing(0, 2);
    
    std::string result(word);
    for(auto& ch : result) {
        const auto& tVariants = this->operator[](ch);
        if(tVariants.empty() || changing(rng) != 0) continue;

        char replacement = tVariants[0];
        for (unsigned i = 0; i < tVariants.size() && changing(rng) != 0; ++i)
            replacement = tVariants[i];

        ch = replacement;
    }

    return result;
}

void MutationDictionary::show() const {
    Console::cout << Console::endl << "Using such password mutations:" << Console::endl;
    for(char letter = char('A'); letter != char('Z'); ++letter) {
        const auto& tVariants = this->operator[](letter);
        if(!tVariants.empty())
            Console::cout << letter << ": " << tVariants << Console::endl;
    }
    Console::cout << Console::endl;
}

const std::string& MutationDictionary::operator[](char key) const {
    if(std::tolower(key) < 'a' || std::tolower(key) > 'z')
        return emptyWord;
    return get()[std::tolower(key) - 'a'];
}

std::optional<std::string> MutationDictionary::backtracking(const std::string& candidate, const std::string& pattern, const Comparator& func) const {
    std::stack<std::pair<char, int>> buffer;
    buffer.push({candidate[0], -1 });

    while(!buffer.empty()) {
        if(buffer.size() >= candidate.size()) {
            const auto string = stackToString(buffer);
            if(func(string, pattern)) return { string };

            unsigned nextPosition = 0;
            do {
                nextPosition = buffer.top().second + 1;
                buffer.pop();

                const auto& variants = getVariants(candidate[buffer.size()]);
                if(nextPosition < variants.size()) break;
            } while (!buffer.empty());

            const auto& variants = getVariants(candidate[buffer.size()]);
            if(nextPosition < variants.size() || !buffer.empty())
                buffer.push({variants[nextPosition], nextPosition});
        } else
            buffer.push({ candidate[buffer.size()], -1 });
    }

    return {};
}

