#include "ReplacementDictionary.h"

const IDictionary::WordArray &ReplacementDictionary::get() const {
    static std::vector<std::string> words = {
            { "4@^да" }, { "8ßв" }, { "[<(с" }, { "д" }, { "3&£е€" },
            { "ƒv" }, { "6&9" }, { "#н" }, { "1|!" }, { "]" },
            { "к<" }, { "!12£7|" }, { "м" }, { "^ทп" }, { "0Øо" },
            { "9р" }, { "20&9" }, { "972®я" }, { "3$z§2" }, { "т7+†" },
            { "vบ" }, { "" }, { "พ" }, { "×" }, { "jу¥" }, { "27s" }
    };

    return words;
}

bool ReplacementDictionary::nextPermutation(const std::string& candidate,
                     const std::string& pattern, std::string& buffer, const Comparator& func) const {
    /* Recursion out. */
    if(buffer.size() == candidate.size())
        return func(buffer, pattern);

    /* Recursion continue. */
    const unsigned position = buffer.size();
    buffer.push_back(candidate[position]);
    if(nextPermutation(candidate, pattern, buffer, func)) return true;
    buffer.pop_back();

    const auto& tVariants = this->operator[](candidate[position]);
    for(char tVariant : tVariants) {
        buffer.push_back(tVariant);
        if(nextPermutation(candidate, pattern, buffer, func)) return true;
        buffer.pop_back();
    }
    return false;
}

std::optional<std::string>
ReplacementDictionary::enumerate(const std::string &candidate, const std::string &pattern,
                                 const ReplacementDictionary::Comparator &func) const {
    std::string buffer; buffer.reserve(candidate.size());
    if(nextPermutation(candidate, pattern, buffer, func))
        return { buffer };
    return {};
}

std::string ReplacementDictionary::rearrange(const std::string& word) const {
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

void ReplacementDictionary::show() const {
    Console::cout << "Using such password mutations:" << Console::endl;
    for(char letter = char('A'); letter != char('Z'); ++letter) {
        const auto& tVariants = this->operator[](letter);
        if(!tVariants.empty())
            Console::cout << letter << ": " << tVariants << Console::endl;
    }
    Console::cout << Console::endl;
}

const std::string &ReplacementDictionary::operator[](char key) const {
    if(std::tolower(key) < 'a' || std::tolower(key) > 'z')
        return emptyWord;
    return get()[std::tolower(key) - 'a'];
}

