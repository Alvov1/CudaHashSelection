#include "MutationDictionary.h"

const IDictionary::WordArray &MutationDictionary::get() const {
    static std::vector<std::string> words = {
            /* A */ { "4@^" }, /* B */ { "86" }, /* C */ { "[<(" }, /* D */ { "" }, /* E */ { "3&" },
            /* F */ { "v" }, /* G */ { "6&9" }, /* H */ { "#" }, /* I */ { "1|/\\!" }, /* J */ { "]}" },
            /* K */ { "(<x" }, /* L */ { "!127|" }, /* M */ { "" }, /* N */ { "^" }, /* O */ { "0" },
            /* P */ { "9?" }, /* Q */ { "20&9" }, /* R */ { "972" }, /* S */ { "3$z2" }, /* T */ { "7+" },
            /* U */ { "v" }, /* V */ { "u" }, /* W */ { "v" }, /* X */ { "%" }, /* Y */ { "j" }, /* Z */ { "27s" }
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
    for(char letter = char('A'); letter <= char('Z'); ++letter) {
        const auto& tVariants = this->operator[](letter);
        if(!tVariants.empty()) {
            Console::cout << letter << ": " << tVariants << Console::endl;

            for(auto& v: tVariants)
                std::cout << static_cast<unsigned>(v) << " ";
            std::cout << std::endl << std::endl;
        }
    }
    Console::cout << Console::endl;
}

const std::string& MutationDictionary::operator[](char key) const {
    if(std::tolower(key) < 'a' || std::tolower(key) > 'z')
        return emptyWord;
    return get()[std::tolower(key) - 'a'];
}

std::optional<std::string> MutationDictionary::backtracking(const std::string& candidate, const std::string& pattern, const Comparator& func) const {
    std::stack<std::pair<char, int>> stack;
    stack.push({candidate[0], -1 });

    while(!stack.empty()) {
        if(stack.size() >= candidate.size()) {
            const auto string = stackToString(stack);
            if(func(string, pattern)) return { string };

            unsigned nextPosition = 0;
            do {
                nextPosition = stack.top().second + 1;
                stack.pop();

                const auto& variants = getVariants(candidate[stack.size()]);
                if(nextPosition < variants.size()) break;
            } while (!stack.empty());

            const auto& variants = getVariants(candidate[stack.size()]);
            if(nextPosition < variants.size() || !stack.empty())
                stack.push({variants[nextPosition], nextPosition});
        } else
            stack.push({candidate[stack.size()], -1 });
    }

    return {};
}

