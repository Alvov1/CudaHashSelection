#ifndef PROJECT2_REPLACEMENTDICTIONARY_H
#define PROJECT2_REPLACEMENTDICTIONARY_H

#include <stack>
#include <string>
#include <random>
#include <optional>
#include <iostream>

#include "Timer.h"
#include "Console.h"
#include "Dictionary.h"

class MutationDictionary final: public IDictionary {
    static std::string stackToString(std::stack<std::pair<char, int>> values) {
        std::string result(values.size(), '\0');
        for(unsigned i = values.size(); !values.empty(); values.pop(), i--)
            result[i - 1] = values.top().first;
        return result;
    }
public:
    const WordArray& get() const override;
    void show() const;
    const std::string& operator[](char key) const;
    const std::string& getVariants(char key) const { return this->operator[](key); }

    std::string mutate(const std::string& word) const;

    std::optional<std::string> backtracking(const std::string& candidate,
            const std::string& pattern, const Comparator& func) const;
};

#endif //PROJECT2_REPLACEMENTDICTIONARY_H
